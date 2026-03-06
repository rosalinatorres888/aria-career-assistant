#!/usr/bin/env python3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml

# Load config
with open('config/aria_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create email
msg = MIMEMultipart()
msg['From'] = config['email']['username']
msg['To'] = config['email']['username']
msg['Subject'] = "[ARIA] 🔥 EMAIL TEST - Jan 15, 2026"

body = """
<html><body>
<h2>🎯 ARIA Email System Test</h2>
<p><strong>Status:</strong> ✅ Email notifications WORKING!</p>
<p><strong>Test Time:</strong> Just now</p>
<p>If you see this, ARIA can send you job alerts! 🚀</p>
</body></html>
"""

msg.attach(MIMEText(body, 'html'))

# Send
print("📧 Sending test email...")
with smtplib.SMTP(config['email']['smtp_server'], config['email']['port']) as server:
    server.starttls()
    server.login(config['email']['username'], config['email']['password'])
    server.send_message(msg)
    
print("✅ EMAIL SENT! Check your inbox: " + config['email']['username'])
