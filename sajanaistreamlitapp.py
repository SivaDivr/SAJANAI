# sajan_ai_gmail.py
import os
import base64
import pandas as pd
import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pathlib import Path
import json
import re
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Load env
load_dotenv()

# ---------- CONFIG ----------
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send"
]
# MODEL = "gemini-2.5-pro"
# API_KEY = os.getenv("GEMINI_API_KEY")

import streamlit as st
API_KEY = st.secrets["GEMINI_API_KEY"]


client = genai.Client(api_key=API_KEY)

# ---------- GMAIL HELPERS ----------
# def gmail_authenticate():
#     creds = None
#     if os.path.exists("token.json"):
#         creds = Credentials.from_authorized_user_file("token.json", SCOPES)
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 "credentials.json", SCOPES
#             )
#             creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
#         with open("token.json", "w") as token:
#             token.write(creds.to_json())
#     return build("gmail", "v1", credentials=creds)

#FOR STREAMLIT DEPLOYMENT
def gmail_authenticate():
    creds = None

    # Load token.json from secrets if it exists
    token_json = st.secrets.get("GMAIL_TOKEN_JSON")
    if token_json:
        creds = Credentials.from_authorized_user_info(json.loads(token_json), SCOPES)

    # If no valid credentials, create new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Load credentials.json from secrets
            credentials_info = st.secrets["GMAIL_CREDENTIALS_JSON"]
            flow = InstalledAppFlow.from_client_config(credentials_info, SCOPES)
            creds = flow.run_console()  # Use console flow in cloud environment

        # Save token back to Streamlit secrets (optional)
        st.session_state["GMAIL_TOKEN_JSON"] = creds.to_json()

    # Build Gmail service
    service = build("gmail", "v1", credentials=creds)
    return service

def get_unread_emails(service, max_results=10):
    results = service.users().messages().list(
        userId="me", labelIds=["INBOX"], q="is:unread", maxResults=max_results
    ).execute()
    msgs = results.get("messages", [])
    emails = []
    for m in msgs:
        msg = service.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["Subject","From","Date"]
        ).execute()
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        emails.append({
            "id": m["id"],
            "threadId": msg["threadId"],
            "snippet": msg["snippet"],
            "headers": headers
        })
    return emails

def extract_text_from_parts(parts):
    text = ""
    for part in parts:
        if part["mimeType"] == "text/plain" and "data" in part["body"]:
            text += base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore") + "\n"
        elif part["mimeType"].startswith("multipart/"):
            text += extract_text_from_parts(part.get("parts", []))
    return text

def get_email_thread(service, thread_id):
    thread = service.users().threads().get(userId="me", id=thread_id, format="full").execute()
    messages = thread.get("messages", [])
    gemini_parts = []
    all_text = []
    included_attachments = []

    for msg in messages:
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
        sender = headers.get("From", "")
        date = headers.get("Date", "")
        subject = headers.get("Subject", "")

        body = ""
        payload = msg["payload"]
        if "parts" in payload:
            body = extract_text_from_parts(payload["parts"])
        elif "data" in payload.get("body", {}):
            body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")

        block = f"From: {sender}\nDate: {date}\nSubject: {subject}\n\n{body.strip()}"
        all_text.append(block)
        gemini_parts.append(types.Part(text=block))

        if "parts" in payload:
            for part in payload["parts"]:
                filename = part.get("filename")
                if filename and part["body"].get("attachmentId"):
                    att_id = part["body"]["attachmentId"]
                    att = service.users().messages().attachments().get(
                        userId="me", messageId=msg["id"], id=att_id
                    ).execute()
                    file_data = base64.urlsafe_b64decode(att["data"].encode("UTF-8"))

                    attachment_part = types.Part(
                        inline_data=types.Blob(
                            mime_type=part.get("mimeType", "application/octet-stream"),
                            data=file_data
                        )
                    )
                    gemini_parts.append(attachment_part)
                    included_attachments.append(filename)

    return "\n\n" + ("-"*80 + "\n\n").join(all_text), gemini_parts, included_attachments

def get_message_id(service, msg_id):
    msg = service.users().messages().get(
        userId="me", id=msg_id,
        format="metadata", metadataHeaders=["Message-ID"]
    ).execute()
    headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
    return headers.get("Message-ID")

def send_reply(service, thread_id, to_addr, subject, draft_text, original_msg_id=None):
    message = MIMEText(draft_text)
    message["to"] = to_addr
    message["subject"] = "Re: " + subject
    if original_msg_id:
        message["In-Reply-To"] = original_msg_id
        message["References"] = original_msg_id

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    result = service.users().messages().send(
        userId="me",
        body={"raw": raw_message, "threadId": thread_id}
    ).execute()
    return result

def mark_as_read(service, msg_id):
    service.users().messages().modify(
        userId="me", id=msg_id,
        body={"removeLabelIds": ["UNREAD"]}
    ).execute()

# ---------- GEMINI HELPERS ----------
def extract_with_gemini(parts):
    prompt = """
    You are an expert in spice extraction company product analysis.
    Extract requirement-related data only: product, form, CU, capsaicin %, origin, notes.
    Ignore promotional material.
    Return JSON with fields: Product, Form, CU (int), CapsaicinPercent (float or null), Origin, Notes.
    """
    resp = client.models.generate_content(
        model=MODEL,
        contents=[types.Part(text=prompt)] + parts
    )
    return resp.text

def safe_parse_extracted(extracted):
    if not extracted:
        return {"Product": None, "Form": None, "CU": None, "CapsaicinPercent": None, "Origin": None, "Notes": ""}
    try:
        parsed = json.loads(extracted)
        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"```json(.*?)```", extracted, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, list) and len(parsed) > 0:
                parsed = parsed[0]
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {"Product": None, "Form": None, "CU": None, "CapsaicinPercent": None, "Origin": None, "Notes": extracted}

# ---------- VECTOR DB & PIS MATCH ----------
encoder = SentenceTransformer('all-MiniLM-L6-v2')

def create_index():
    return faiss.IndexFlatL2(384)

def get_pis_embedding(pis_entry):
    text = f"{pis_entry['title']} {pis_entry['form']} CU:{pis_entry['cu_min']}-{pis_entry['cu_max']}"
    if pis_entry['capsaicin_percent_min'] is not None:
        text += f" Capsaicin:{pis_entry['capsaicin_percent_min']}%"
    if pis_entry['origin'] != "Any":
        text += f" Origin:{pis_entry['origin']}"
    return encoder.encode([text])[0]

def create_vector_db(df_pis):
    index = create_index()
    vectors = np.array([get_pis_embedding(row) for _, row in df_pis.iterrows()])
    if len(vectors) > 0:
        index.add(vectors.astype('float32'))
    return index

def load_pis(pis_path="pis_demo.csv"):
    if Path(pis_path).exists():
        df_pis = pd.read_csv(pis_path)
    else:
        df_pis = pd.DataFrame([
            {"pis_id":"4010000334","title":"PAPRIKA OR ENCAP 25000-27500CU W/S","form":"powder","cu_min":25500,"cu_max":27500,"capsaicin_percent_min":None,"capsaicin_percent_max":None,"origin":"Any"},
            {"pis_id":"4000000937","title":"PAPRIKA OR 40 CU / 2.5% Capsaicin","form":"oleoresin","cu_min":40,"cu_max":40,"capsaicin_percent_min":2.5,"capsaicin_percent_max":2.5,"origin":"Any"},
            {"pis_id":"4010009475","title":"PAPRIKA OR STANDARD 100000 CU with 300-800 ppm capsaicin","form":"oleoresin","cu_min":100000,"cu_max":100000,"capsaicin_percent_min":None,"capsaicin_percent_max":None,"origin":"Any"}
        ])
    return df_pis

def match_pis(req, pis_df):
    if pis_df.empty:
        return None, []
    query_text = f"{req.get('Product','')} {req.get('Form','')}"
    if req.get('CU'): query_text += f" CU:{req.get('CU')}"
    if req.get('CapsaicinPercent'): query_text += f" Capsaicin:{req.get('CapsaicinPercent')}%"
    if req.get('Origin'): query_text += f" Origin:{req.get('Origin')}"

    query_vector = encoder.encode([query_text])[0].astype('float32').reshape(1, -1)
    index = create_vector_db(pis_df)
    k = min(5, len(pis_df))
    if k == 0: return None, []
    distances, indices = index.search(query_vector, k)
    best_idx = indices[0][0]
    best_distance = distances[0][0]
    row = pis_df.iloc[best_idx]
    max_distance = np.max(distances)
    score = 1 - (best_distance / max_distance if max_distance > 0 else 0)
    return {
        "pis_id": row["pis_id"],
        "title": row["title"],
        "score": round(score, 2),
        "shortlist": pis_df.iloc[indices[0]].to_dict(orient="records")
    }, pis_df[["pis_id","title"]].to_dict(orient="records")

# ---------- LLM MATCH ----------
def llm_match_pis(req, pis_df):
    prompt = f"""
    You are a product matching assistant.
    Given a requirement and possible PIS options, select the best match.
    Consider product name, CU (Color Units), capsaicin %, origin, and form (powder, oleoresin, etc.).
    - If the requested CU is slightly outside the PIS CU range, mention it as 'close to range' instead of 'within range'.
    Respond in JSON as:
    {{
        "pis_id": "<best id or null>",
        "title": "<best title or null>",
        "reason": "<why selected>"
    }}
    Requirement:
    {json.dumps(req, indent=2)}

    PIS entries:
    {pis_df.to_dict(orient='records')}
    """
    resp = client.models.generate_content(model=MODEL, contents=[types.Part(text=prompt)])
    try:
        text = resp.text
        match = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        return match
    except Exception:
        return None

# # ---------- DRAFT REPLY ----------
# def draft_reply(req, best_match=None):
#     product = req.get("Product", "the requested item") if isinstance(req, dict) else "the requested item"
#     details = []
#     if req.get("Form"): details.append(req.get("Form"))
#     if req.get("CU"): details.append(f"{req.get('CU')} CU")
#     if req.get("CapsaicinPercent"): details.append(f"{req.get('CapsaicinPercent')}% capsaicin")
#     if req.get("Origin"): details.append(f"Origin:{req.get('Origin')}")
#     details_str = ", ".join(details) if details else ""

#     if best_match:
#         message = f"""
# Dear {req.get("ContactName","Sir/Madam")},

# Thank you for your enquiry regarding {product}{f' ({details_str})' if details_str else ''}.

# Based on your requirement, our system has identified the most relevant Product Information Sheet (PIS):

# **PIS ID:** {best_match.get('pis_id')}
# **Product Title:** {best_match.get('title')}
# **Match Confidence:** {best_match.get('score',0):.2f}

# Kindly review the attached PIS and confirm if this aligns with your requirement.
# If not, please provide additional details so we can refine the match.

# Best regards,
# Siva
# AI Product Information Assistant
# Synthite Industries Pvt. Ltd.
#         """
#     else:
#         message = f"""
# Dear {req.get("ContactName","Sir/Madam")},

# Thank you for your enquiry regarding {product}{f' ({details_str})' if details_str else ''}.

# We could not identify a close PIS match from our database based on the available details.
# Kindly share the specific CU, capsaicin %, or product form so we can assist you further.

# Best regards,
# Siva
# AI Product Information Assistant
# Synthite Industries Pvt. Ltd.
#         """
#     return message.strip()
def draft_reply(req, best_match=None):
    print("<<DRAFT REPLY CALLED>>")
    print("<<REQ>>",req)
    print("<<BEST MATCH>>",best_match)
    product = req.get("Product", "the requested item") if isinstance(req, dict) else "the requested item"
    details = []
    if req.get("Form"): details.append(req.get("Form"))
    if req.get("CU"): details.append(f"{req.get('CU')} CU")
    if req.get("CapsaicinPercent"): details.append(f"{req.get('CapsaicinPercent')}% capsaicin")
    if req.get("Origin"): details.append(f"Origin:{req.get('Origin')}")
    details_str = ", ".join(details) if details else ""
    print(details_str,"<<DETAILS STR>>")

    if best_match:
        score = best_match.get('score', 0)
        # Determine handling note based on match score
        if score < 0.30:
            handling_note = "\n\n⚠️ This product appears to be new. Our NPD team will handle creating the item."
        elif 0.30 <= score < 0.90:
            handling_note = "\n\n✅ Our CVJ team can handle this product."
        else:
            handling_note = ""   

        message = f"""
Dear {req.get("ContactName","Sir/Madam")},

Thank you for your enquiry regarding {product}{f' ({details_str})' if details_str else ''}.

Based on your requirement, our system has identified the most relevant Product Information Sheet (PIS):

**PIS ID:** {best_match.get('pis_id')}
**Product Title:** {best_match.get('title')}
**Match Confidence:** {score:.2f}{handling_note}

Kindly review the PIS details and confirm if this aligns with your requirement.


Best regards,
Siva
AI Product Information Assistant

        """
    else:
        message = f"""
Dear {req.get("ContactName","Sir/Madam")},

Thank you for your enquiry regarding {product}{f' ({details_str})' if details_str else ''}.

We could not identify a close PIS match from our database based on the available details.
Kindly share the specific CU, capsaicin %, or product form so we can assist you further.
⚠️ This product appears to be new. Our NPD team will handle creating the item.

Best regards,
Siva
AI Product Information Assistant
        """
    return message.strip()


# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="Sajan AI - Gmail Assistant", layout="wide")
st.title("📧 Sajan AI - Gmail Integrated Assistant")

service = gmail_authenticate()
emails = get_unread_emails(service, max_results=10)

st.sidebar.header("Unread Emails")
if not emails:
    st.sidebar.write("✅ No unread emails")
else:
    subject_list = [f"{e['headers'].get('Subject','(No Subject)')} — {e['headers'].get('From','')}" for e in emails]
    selected_subject = st.sidebar.radio("Select email", subject_list)
    selected_index = subject_list.index(selected_subject)
    selected_email = emails[selected_index]
    st.session_state["selected_email"] = selected_email

# Initialize session state containers
st.session_state.setdefault("req", None)
st.session_state.setdefault("faiss_result", None)
st.session_state.setdefault("llm_result", None)
st.session_state.setdefault("best_match", None)
st.session_state.setdefault("draft_text", None)
st.session_state.setdefault("pis_df", load_pis())
st.session_state.setdefault("included_attachments", None)
st.session_state.setdefault("gemini_parts", None)
st.session_state.setdefault("email_body", None)

# Show email & auto-load thread
sel = st.session_state["selected_email"]
st.subheader(f"📌 {sel['headers'].get('Subject')}")
st.caption(f"From: {sel['headers'].get('From')} | Date: {sel['headers'].get('Date')}")
st.write(f"Snippet: {sel['snippet']}")

# Auto-fetch full thread + attachments
if st.session_state.get("email_body") is None:
    with st.spinner("Loading full email thread & attachments..."):
        body, gemini_parts, included_attachments = get_email_thread(service, sel["threadId"])
        st.session_state["email_body"] = body
        st.session_state["gemini_parts"] = gemini_parts
        st.session_state["included_attachments"] = included_attachments

st.text_area("📜 Email Thread", st.session_state["email_body"], height=300)
if st.session_state.get("included_attachments"):
    st.info(f"📎 Attachments considered: {', '.join(st.session_state['included_attachments'])}")

# Extract requirement automatically
if st.session_state.get("req") is None:
    with st.spinner("Extracting requirement from thread..."):
        extracted = extract_with_gemini(st.session_state["gemini_parts"])
        st.session_state["req"] = safe_parse_extracted(extracted)

st.json(st.session_state["req"])

# Run hybrid matching automatically
if st.button("🔍 Run Matching"):
    req = st.session_state["req"]
    pis_df = st.session_state["pis_df"]
    st.session_state["faiss_result"], _ = match_pis(req, pis_df)

    # Hybrid LLM on FAISS shortlist
    shortlist = st.session_state["faiss_result"].get("shortlist", []) if st.session_state.get("faiss_result") else []
    if len(shortlist) > 0:
        top_df = pd.DataFrame(shortlist)
        with st.spinner("Running LLM on FAISS shortlist..."):
            llm_res = llm_match_pis(req, top_df)
            st.session_state["llm_result"] = llm_res
            if llm_res and llm_res.get("pis_id"):
                st.success(f"Hybrid LLM matched: {llm_res['pis_id']} — {llm_res['title']}")
                st.write("LLM reason:")
                st.write(llm_res.get("reason", "No reason provided."))
                st.session_state["best_match"] = {"pis_id": llm_res["pis_id"], "title": llm_res["title"], "score": 0.98}
            else:
                st.warning("LLM did not choose any shortlist entries.")
    else:
        st.warning("No FAISS shortlist to run LLM on.")

# Draft reply
if st.session_state.get("best_match") or st.session_state.get("faiss_result"):
    if st.session_state.get("best_match"):
        st.subheader("🏷️ Match Result")
        match = st.session_state.get("best_match") or st.session_state.get("faiss_result")
        st.write(f"**PIS ID:** {match.get('pis_id')}")
        st.write(f"**Title:** {match.get('title')}")
        if "score" in match:
            st.write(f"**Match Score:** {match.get('score'):.2f}")
    auto_draft = draft_reply(st.session_state["req"], st.session_state.get("best_match"))
    if st.session_state.get("draft_text") is None or st.session_state.get("draft_text") != auto_draft:
        st.session_state["draft_text"] = auto_draft

    st.subheader("✉️ Draft Reply (editable)")
    st.session_state["draft_text"] = st.text_area("Edit draft", value=st.session_state["draft_text"], height=220)

    # Approve & Send
    st.session_state["to_addr"] = sel["headers"].get("From")
    st.session_state["thread_id"] = sel["threadId"]
    st.session_state["msg_id"] = sel["id"]
    st.session_state["subject"] = sel["headers"].get("Subject")

    if st.button("✅ Approve & Send"):
        try:
            original_msg_id = get_message_id(service, st.session_state["msg_id"])
        except Exception as e:
            original_msg_id = None
            st.warning(f"Could not fetch original Message-ID: {e}")

        try:
            result = send_reply(
                service,
                st.session_state["thread_id"],
                st.session_state["to_addr"],
                st.session_state["subject"],
                st.session_state["draft_text"],
                original_msg_id
            )
            try:
                mark_as_read(service, st.session_state["msg_id"])
            except Exception as e:
                st.warning(f"Sent but couldn't mark as read: {e}")
            st.success("📩 Reply sent ✅")
            st.json(result)
            for k in ["req", "faiss_result", "llm_result", "best_match", "draft_text", "gemini_parts", "included_attachments", "email_body"]:
                st.session_state.pop(k, None)
        except Exception as e:
            st.error(f"❌ Error while sending: {e}")
