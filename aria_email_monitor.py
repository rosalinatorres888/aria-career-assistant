"""
ARIA Email Job Monitor
Scans Gmail via IMAP for job alert emails from LinkedIn, Indeed, Glassdoor
Parses job details and feeds them into ARIA's pipeline
"""

import imaplib
import email
from email.header import decode_header
import re
import os
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger("ARIA.email_monitor")

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from aria_integrations import JobPosting, ApplicationTracker
except ImportError:
    JobPosting = None


class EmailJobMonitor:
    """
    Monitor Gmail inbox for job alert emails from LinkedIn, Indeed, Glassdoor.
    Parses job postings from email content and returns normalized results.

    Setup:
      1. Enable IMAP in Gmail: Settings > See all settings > Forwarding and POP/IMAP > Enable IMAP
      2. Use a Gmail App Password (not your regular password)
      3. Set env vars:
           ARIA_EMAIL_USER=os.environ.get('ARIA_EMAIL_USER', 'your@gmail.com')
           ARIA_EMAIL_PASS=your-16-char-app-password
    """

    def __init__(self):
        self.email_address = os.environ.get("ARIA_EMAIL_USER", "")
        self.app_password = os.environ.get("ARIA_EMAIL_PASS", "")
        self.imap_server = "imap.gmail.com"
        self.imap_port = 993
        self._mail = None
        self._seen_ids = set()

        # Job alert senders we look for
        self.alert_senders = {
            "linkedin": [
                "jobs-noreply@linkedin.com",
                "linkedin@e.linkedin.com",
                "invitations@linkedin.com",
            ],
            "indeed": [
                "alert@indeed.com",
                "noreply@indeed.com",
                "jobalerts-noreply@indeed.com",
            ],
            "glassdoor": [
                "noreply@glassdoor.com",
                "alerts@glassdoor.com",
            ],
            "wellfound": [
                "noreply@wellfound.com",
                "notifications@angel.co",
            ],
            "builtin": [
                "notifications@builtin.com",
                "noreply@builtin.com",
            ],
        }

    @property
    def is_configured(self) -> bool:
        return bool(self.email_address and self.app_password)

    def _connect(self) -> bool:
        """Connect to Gmail IMAP."""
        try:
            self._mail = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            self._mail.login(self.email_address, self.app_password)
            logger.info(f"IMAP connected to {self.email_address}")
            return True
        except Exception as e:
            logger.error(f"IMAP connection failed: {e}")
            self._mail = None
            return False

    def _disconnect(self):
        """Close IMAP connection."""
        if self._mail:
            try:
                self._mail.close()
                self._mail.logout()
            except Exception:
                pass
            self._mail = None

    async def scan_inbox(self, hours_back: int = 24) -> List[Dict]:
        """
        Scan Gmail inbox for job alert emails from the last N hours.
        Returns list of parsed job postings.
        """
        if not self.is_configured:
            logger.warning("Email monitor not configured — set ARIA_EMAIL_USER and ARIA_EMAIL_PASS")
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._scan_inbox_sync, hours_back)

    def _scan_inbox_sync(self, hours_back: int) -> List[Dict]:
        """Synchronous inbox scan (runs in thread executor)."""
        if not self._connect():
            return []

        jobs = []
        try:
            self._mail.select("INBOX")

            # Search for recent emails from job alert senders
            since_date = (datetime.now() - timedelta(hours=hours_back)).strftime("%d-%b-%Y")

            # Build search queries for each source
            all_senders = []
            for source_senders in self.alert_senders.values():
                all_senders.extend(source_senders)

            for sender in all_senders:
                try:
                    search_query = f'(SINCE {since_date} FROM "{sender}")'
                    status, message_ids = self._mail.search(None, search_query)

                    if status != "OK" or not message_ids[0]:
                        continue

                    ids = message_ids[0].split()
                    logger.info(f"  Found {len(ids)} emails from {sender}")

                    for msg_id in ids[-20:]:  # Cap at 20 per sender
                        try:
                            status, msg_data = self._mail.fetch(msg_id, "(RFC822)")
                            if status != "OK":
                                continue

                            raw_email = msg_data[0][1]
                            msg = email.message_from_bytes(raw_email)

                            parsed_jobs = self._parse_email(msg, sender)
                            jobs.extend(parsed_jobs)

                        except Exception as e:
                            logger.debug(f"Error parsing email {msg_id}: {e}")

                except Exception as e:
                    logger.debug(f"Search error for {sender}: {e}")

            # Also search by subject keywords
            keyword_searches = [
                f'(SINCE {since_date} SUBJECT "job alert")',
                f'(SINCE {since_date} SUBJECT "new jobs for you")',
                f'(SINCE {since_date} SUBJECT "jobs you may be interested")',
                f'(SINCE {since_date} SUBJECT "recommended jobs")',
                f'(SINCE {since_date} SUBJECT "new job matches")',
            ]

            for search_query in keyword_searches:
                try:
                    status, message_ids = self._mail.search(None, search_query)
                    if status != "OK" or not message_ids[0]:
                        continue

                    ids = message_ids[0].split()
                    for msg_id in ids[-10:]:
                        try:
                            status, msg_data = self._mail.fetch(msg_id, "(RFC822)")
                            if status != "OK":
                                continue
                            raw_email = msg_data[0][1]
                            msg = email.message_from_bytes(raw_email)
                            parsed_jobs = self._parse_email(msg, "keyword_match")
                            jobs.extend(parsed_jobs)
                        except Exception:
                            continue
                except Exception:
                    continue

        except Exception as e:
            logger.error(f"Inbox scan error: {e}")
        finally:
            self._disconnect()

        # Deduplicate
        unique = self._deduplicate(jobs)
        logger.info(f"Email scan complete: {len(unique)} unique jobs from {len(jobs)} total parsed")
        return unique

    def _parse_email(self, msg, sender: str) -> List[Dict]:
        """Parse a single email message for job postings."""
        jobs = []

        # Get email body
        body_html = ""
        body_text = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        decoded = payload.decode(charset, errors="replace")
                        if content_type == "text/html":
                            body_html = decoded
                        elif content_type == "text/plain":
                            body_text = decoded
                except Exception:
                    continue
        else:
            try:
                payload = msg.get_payload(decode=True)
                charset = msg.get_content_charset() or "utf-8"
                decoded = payload.decode(charset, errors="replace")
                if msg.get_content_type() == "text/html":
                    body_html = decoded
                else:
                    body_text = decoded
            except Exception:
                pass

        # Determine source
        source = "Email Alert"
        sender_lower = sender.lower()
        for src_name, senders in self.alert_senders.items():
            if any(s in sender_lower for s in senders):
                source = src_name.capitalize()
                break

        # Parse based on source
        if "linkedin" in sender_lower:
            jobs = self._parse_linkedin_email(body_html or body_text, source)
        elif "indeed" in sender_lower:
            jobs = self._parse_indeed_email(body_html or body_text, source)
        elif "glassdoor" in sender_lower:
            jobs = self._parse_glassdoor_email(body_html or body_text, source)
        else:
            jobs = self._parse_generic_email(body_html or body_text, source)

        return jobs

    def _parse_linkedin_email(self, body: str, source: str) -> List[Dict]:
        """Parse LinkedIn job alert email."""
        jobs = []
        if not BeautifulSoup or not body:
            return jobs

        soup = BeautifulSoup(body, "html.parser")

        # LinkedIn job alerts typically have job cards with links
        # Pattern 1: Links containing /jobs/view/
        job_links = soup.find_all("a", href=re.compile(r"linkedin\.com/jobs/view|linkedin\.com/comm/jobs"))

        for link in job_links:
            href = link.get("href", "")
            text = link.get_text(strip=True)

            if not text or len(text) < 5:
                continue

            # Try to extract title and company
            # LinkedIn emails usually have "Title at Company" or "Title - Company"
            title, company, location = self._extract_job_details(text, link)

            if title:
                job_id = hashlib.md5(f"linkedin_email:{title}:{company}".encode()).hexdigest()[:12]
                jobs.append({
                    "id": job_id,
                    "title": title,
                    "company": company,
                    "location": location,
                    "url": self._clean_url(href),
                    "source": "LinkedIn (Email)",
                    "posted_date": datetime.now().isoformat(),
                })

        # Pattern 2: Look for structured sections
        sections = soup.find_all("td") + soup.find_all("div")
        for section in sections:
            text = section.get_text(strip=True)
            # Look for patterns like "Software Engineer\nCompany Name\nLocation"
            if re.search(r"(engineer|analyst|scientist|developer|intern|co-op)", text, re.I):
                link = section.find("a", href=True)
                if link:
                    title, company, location = self._extract_job_details(text, section)
                    if title and len(title) > 5:
                        job_id = hashlib.md5(f"linkedin_email2:{title}:{company}".encode()).hexdigest()[:12]
                        jobs.append({
                            "id": job_id,
                            "title": title,
                            "company": company,
                            "location": location,
                            "url": self._clean_url(link.get("href", "")),
                            "source": "LinkedIn (Email)",
                            "posted_date": datetime.now().isoformat(),
                        })

        return jobs

    def _parse_indeed_email(self, body: str, source: str) -> List[Dict]:
        """Parse Indeed job alert email."""
        jobs = []
        if not BeautifulSoup or not body:
            return jobs

        soup = BeautifulSoup(body, "html.parser")

        # Indeed alerts have links to job postings
        job_links = soup.find_all("a", href=re.compile(r"indeed\.com.*jk=|indeed\.com/viewjob|indeed\.com/rc/clk"))

        for link in job_links:
            href = link.get("href", "")
            text = link.get_text(strip=True)

            if not text or len(text) < 5:
                # Try parent element
                parent = link.parent
                if parent:
                    text = parent.get_text(strip=True)

            title, company, location = self._extract_job_details(text, link)

            if title:
                job_id = hashlib.md5(f"indeed_email:{title}:{company}".encode()).hexdigest()[:12]
                jobs.append({
                    "id": job_id,
                    "title": title,
                    "company": company,
                    "location": location,
                    "url": self._clean_url(href),
                    "source": "Indeed (Email)",
                    "posted_date": datetime.now().isoformat(),
                })

        return jobs

    def _parse_glassdoor_email(self, body: str, source: str) -> List[Dict]:
        """Parse Glassdoor job alert email."""
        jobs = []
        if not BeautifulSoup or not body:
            return jobs

        soup = BeautifulSoup(body, "html.parser")
        job_links = soup.find_all("a", href=re.compile(r"glassdoor\.com.*job"))

        for link in job_links:
            href = link.get("href", "")
            text = link.get_text(strip=True)
            title, company, location = self._extract_job_details(text, link)

            if title:
                job_id = hashlib.md5(f"glassdoor_email:{title}:{company}".encode()).hexdigest()[:12]
                jobs.append({
                    "id": job_id,
                    "title": title,
                    "company": company,
                    "location": location,
                    "url": self._clean_url(href),
                    "source": "Glassdoor (Email)",
                    "posted_date": datetime.now().isoformat(),
                })

        return jobs

    def _parse_generic_email(self, body: str, source: str) -> List[Dict]:
        """Parse generic job alert email using pattern matching."""
        jobs = []
        if not body:
            return jobs

        if BeautifulSoup:
            soup = BeautifulSoup(body, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
        else:
            text = body

        # Look for job-like patterns
        patterns = [
            r"(?P<title>(?:Data|ML|AI|Machine Learning|Software|Analytics|Business Intelligence)[\w\s\-/]+(?:Engineer|Analyst|Scientist|Developer|Intern|Co-op))\s*(?:at|@|-)\s*(?P<company>[\w\s&.]+?)(?:\s*[-|]\s*(?P<location>[\w\s,]+))?",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                title = match.group("title").strip()
                company = match.group("company").strip() if match.group("company") else "Unknown"
                location = match.group("location").strip() if match.group("location") else ""

                if title and len(title) > 5:
                    job_id = hashlib.md5(f"generic:{title}:{company}".encode()).hexdigest()[:12]
                    jobs.append({
                        "id": job_id,
                        "title": title,
                        "company": company[:100],
                        "location": location[:100],
                        "url": "",
                        "source": source,
                        "posted_date": datetime.now().isoformat(),
                    })

        return jobs

    def _extract_job_details(self, text: str, element=None):
        """Extract title, company, and location from text."""
        title = ""
        company = "Unknown"
        location = ""

        if not text:
            return title, company, location

        # Clean up text
        text = re.sub(r"\s+", " ", text).strip()

        # Pattern: "Title at Company - Location"
        match = re.match(r"^(.+?)\s+(?:at|@)\s+(.+?)(?:\s*[-–|]\s*(.+))?$", text)
        if match:
            title = match.group(1).strip()
            company = match.group(2).strip()
            location = (match.group(3) or "").strip()
            return title, company, location

        # Pattern: "Title - Company - Location"
        match = re.match(r"^(.+?)\s*[-–]\s*(.+?)(?:\s*[-–]\s*(.+))?$", text)
        if match:
            title = match.group(1).strip()
            company = match.group(2).strip()
            location = (match.group(3) or "").strip()
            return title, company, location

        # Pattern: "Title\nCompany\nLocation" (from parent element)
        if element is not None and BeautifulSoup:
            parts = []
            for child in element.children:
                t = child.string if hasattr(child, "string") and child.string else ""
                if t and t.strip():
                    parts.append(t.strip())

            if not parts:
                parts = [line.strip() for line in element.get_text(separator="\n").split("\n") if line.strip()]

            if len(parts) >= 2:
                title = parts[0]
                company = parts[1]
                if len(parts) >= 3:
                    location = parts[2]
                return title, company, location

        # Fallback: just use the text as title
        if len(text) < 100:
            title = text

        return title, company, location

    @staticmethod
    def _clean_url(url: str) -> str:
        """Clean tracking parameters from URLs."""
        if not url:
            return ""
        # Remove common tracking params
        url = re.split(r"[?&](?:utm_|trk=|refId=|trackingId=)", url)[0]
        return url

    def _deduplicate(self, jobs: List[Dict]) -> List[Dict]:
        """Remove duplicate jobs."""
        seen = set()
        unique = []
        for job in jobs:
            key = f"{job.get('title', '').lower()}|{job.get('company', '').lower()}"
            if key not in seen and job.get("id") not in self._seen_ids:
                seen.add(key)
                self._seen_ids.add(job.get("id", ""))
                unique.append(job)
        return unique
