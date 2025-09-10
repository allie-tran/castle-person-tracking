
import smtplib
import os
from dotenv import load_dotenv
from email.message import EmailMessage

load_dotenv()

EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
ALERT_EMAIL = os.getenv("ALERT_EMAIL")
assert ALERT_EMAIL is not None, "ALERT_EMAIL environment variable not set"


def send_email(subject: str, body: str, to_email: str = ALERT_EMAIL):
    """
    Sends an email with the specified subject and body to the given email address.
    Defaults to ALERT_EMAIL if no recipient is specified.
    """
    try:
        assert EMAIL is not None, "EMAIL environment variable not set"
        assert PASSWORD is not None, "PASSWORD environment variable not set"

        # Create the email message
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = EMAIL
        msg['To'] = to_email
        msg.set_content(body)
        # Send the email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.starttls()
            smtp.login(EMAIL, PASSWORD)
            smtp.send_message(msg)
        print(f"Email sent successfully to {to_email}!")
    except Exception as e:
        print(f"Failed to send email: {e}")
