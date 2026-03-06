"""
ARIA Integrations Module
Real implementations for:
  1. Opportunity Scanner (Indeed, Wellfound, YC, Built In Boston, Greenhouse/Lever)
  2. Application Tracker (MySQL/MongoDB)
  3. Daily Report (rich HTML email)
  4. Email Alerts (Gmail SMTP)
"""

import asyncio
import hashlib
import re
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger("ARIA.integrations")

try:
    import httpx
except ImportError:
    httpx = None
    logger.warning("httpx not installed: pip install httpx")

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    logger.warning("beautifulsoup4 not installed: pip install beautifulsoup4")

try:
    import pymysql
except ImportError:
    pymysql = None

try:
    import pymongo
except ImportError:
    pymongo = None


@dataclass
class JobPosting:
    id: str
    title: str
    company: str
    location: str
    url: str
    source: str
    description: str = ""
    posted_date: Optional[datetime] = None
    salary_range: str = ""
    confidence: float = 0.0
    keywords_matched: List[str] = field(default_factory=list)
    is_new: bool = True

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "title": self.title, "company": self.company,
            "location": self.location, "url": self.url, "source": self.source,
            "description": self.description[:500],
            "posted_date": self.posted_date.isoformat() if self.posted_date else None,
            "salary_range": self.salary_range,
            "confidence": round(self.confidence, 3),
            "keywords_matched": self.keywords_matched, "is_new": self.is_new,
        }


@dataclass
class Application:
    id: str
    job_id: str
    company: str
    role: str
    url: str
    status: str = "identified"
    applied_date: Optional[datetime] = None
    follow_up_date: Optional[datetime] = None
    notes: str = ""
    source: str = ""
    confidence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "job_id": self.job_id, "company": self.company,
            "role": self.role, "url": self.url, "status": self.status,
            "applied_date": self.applied_date.isoformat() if self.applied_date else None,
            "follow_up_date": self.follow_up_date.isoformat() if self.follow_up_date else None,
            "notes": self.notes, "source": self.source,
            "confidence_score": round(self.confidence_score, 3),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class OpportunityScanner:
    def __init__(self, config=None):
        from aria_job_config import (
            TARGET_TITLES, POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS,
            TARGET_LOCATIONS, EXCLUDED_COMPANIES, SCORING,
        )
        self.target_titles = TARGET_TITLES
        self.positive_keywords = POSITIVE_KEYWORDS
        self.negative_keywords = NEGATIVE_KEYWORDS
        self.target_locations = TARGET_LOCATIONS
        self.excluded_companies = EXCLUDED_COMPANIES
        self.scoring = SCORING
        self.config = config or {}
        self._seen_ids = set()
        self._http_client = None

    @property
    def http(self):
        if self._http_client is None and httpx is not None:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
                follow_redirects=True,
            )
        return self._http_client

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()

    async def scan_all(self) -> List[JobPosting]:
        if httpx is None:
            logger.error("httpx required for scanning: pip install httpx")
            return []
        all_jobs = []
        scanners = [
            ("Indeed", self.scan_indeed),
            ("Wellfound", self.scan_wellfound),
            ("YCombinator", self.scan_ycombinator),
            ("BuiltInBoston", self.scan_builtin_boston),
            ("Google Jobs", self.scan_google_jobs),
        ]
        for name, scanner in scanners:
            try:
                jobs = await scanner()
                logger.info(f"  {name}: found {len(jobs)} postings")
                all_jobs.extend(jobs)
            except Exception as e:
                logger.warning(f"  {name} scanner failed: {e}")
        unique_jobs = self._deduplicate(all_jobs)
        for job in unique_jobs:
            job.confidence = self._score_job(job)
        unique_jobs.sort(key=lambda j: j.confidence, reverse=True)
        logger.info(f"Total unique opportunities: {len(unique_jobs)}")
        return unique_jobs

    async def scan_indeed(self) -> List[JobPosting]:
        jobs = []
        queries = ["data+analyst+intern", "machine+learning+intern", "data+engineer+entry+level", "data+science+intern"]
        for query in queries:
            try:
                url = f"https://www.indeed.com/jobs?q={query}&l=Boston%2C+MA&radius=50&fromage=3"
                resp = await self.http.get(url)
                if resp.status_code != 200 or BeautifulSoup is None:
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                cards = soup.select("div.job_seen_beacon, div.jobsearch-ResultsList > div")
                for card in cards[:15]:
                    title_el = card.select_one("h2.jobTitle a, h2.jobTitle span")
                    company_el = card.select_one("span[data-testid='company-name'], span.companyName")
                    location_el = card.select_one("div[data-testid='text-location'], div.companyLocation")
                    link_el = card.select_one("h2.jobTitle a")
                    if not title_el:
                        continue
                    title = title_el.get_text(strip=True)
                    company = company_el.get_text(strip=True) if company_el else "Unknown"
                    location = location_el.get_text(strip=True) if location_el else ""
                    href = link_el.get("href", "") if link_el else ""
                    if href and not href.startswith("http"):
                        href = f"https://www.indeed.com{href}"
                    job_id = hashlib.md5(f"indeed:{title}:{company}".encode()).hexdigest()[:12]
                    jobs.append(JobPosting(id=job_id, title=title, company=company, location=location, url=href, source="Indeed"))
                await asyncio.sleep(1)
            except Exception as e:
                logger.debug(f"Indeed query '{query}' error: {e}")
        return jobs

    async def scan_wellfound(self) -> List[JobPosting]:
        jobs = []
        try:
            resp = await self.http.get("https://wellfound.com/role/l/data-scientist/boston")
            if resp.status_code != 200 or BeautifulSoup is None:
                return jobs
            soup = BeautifulSoup(resp.text, "html.parser")
            for script in soup.select("script[type='application/ld+json']"):
                try:
                    data = json.loads(script.string)
                    if isinstance(data, list):
                        for item in data:
                            if item.get("@type") == "JobPosting":
                                jobs.append(self._parse_ld_json_job(item, "Wellfound"))
                    elif isinstance(data, dict) and data.get("@type") == "JobPosting":
                        jobs.append(self._parse_ld_json_job(data, "Wellfound"))
                except (json.JSONDecodeError, KeyError):
                    continue
        except Exception as e:
            logger.debug(f"Wellfound scan error: {e}")
        return jobs

    async def scan_ycombinator(self) -> List[JobPosting]:
        jobs = []
        try:
            resp = await self.http.get("https://www.workatastartup.com/companies?jobType=fulltime&jobType=intern&roleType=eng&sortBy=created_desc")
            if resp.status_code != 200 or BeautifulSoup is None:
                return jobs
            soup = BeautifulSoup(resp.text, "html.parser")
            for script in soup.select("script[type='application/ld+json']"):
                try:
                    data = json.loads(script.string)
                    if isinstance(data, list):
                        for item in data:
                            if item.get("@type") == "JobPosting":
                                jobs.append(self._parse_ld_json_job(item, "YCombinator"))
                except (json.JSONDecodeError, KeyError):
                    continue
        except Exception as e:
            logger.debug(f"YC scan error: {e}")
        return jobs

    async def scan_builtin_boston(self) -> List[JobPosting]:
        jobs = []
        for path in ["/jobs/data-analytics", "/jobs/machine-learning", "/jobs/data-science"]:
            try:
                resp = await self.http.get(f"https://www.builtinboston.com{path}?page=1")
                if resp.status_code != 200 or BeautifulSoup is None:
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                for card in soup.select("div[class*='job-card'], div[class*='job-bounded']")[:15]:
                    title_el = card.select_one("h2 a, [class*='job-title'] a")
                    company_el = card.select_one("[class*='company-title'], span[class*='company']")
                    if not title_el:
                        continue
                    title = title_el.get_text(strip=True)
                    company = company_el.get_text(strip=True) if company_el else "Unknown"
                    href = title_el.get("href", "")
                    if href and not href.startswith("http"):
                        href = f"https://www.builtinboston.com{href}"
                    job_id = hashlib.md5(f"builtin:{title}:{company}".encode()).hexdigest()[:12]
                    jobs.append(JobPosting(id=job_id, title=title, company=company, location="Boston, MA", url=href, source="BuiltInBoston"))
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.debug(f"Built In Boston error: {e}")
        return jobs

    async def scan_google_jobs(self) -> List[JobPosting]:
        jobs = []
        queries = [
            'site:linkedin.com/jobs "data analyst" "Boston" intern OR "entry level"',
            'site:linkedin.com/jobs "machine learning" "Boston" intern OR junior',
        ]
        for query in queries:
            try:
                resp = await self.http.get(f"https://www.google.com/search?q={query}&tbs=qdr:w")
                if resp.status_code != 200 or BeautifulSoup is None:
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                for result in soup.select("div.g")[:10]:
                    link_el = result.select_one("a[href*='linkedin.com/jobs']")
                    title_el = result.select_one("h3")
                    if not link_el or not title_el:
                        continue
                    title_text = title_el.get_text(strip=True)
                    href = link_el.get("href", "")
                    parts = title_text.split(" - ")
                    title = parts[0].strip()
                    company = parts[1].strip() if len(parts) > 1 else "Via LinkedIn"
                    job_id = hashlib.md5(f"google_li:{title}:{company}".encode()).hexdigest()[:12]
                    jobs.append(JobPosting(id=job_id, title=title, company=company, location="Boston, MA", url=href, source="LinkedIn (via Google)"))
                await asyncio.sleep(2)
            except Exception as e:
                logger.debug(f"Google Jobs error: {e}")
        return jobs

    def _parse_ld_json_job(self, data, source):
        title = data.get("title", "Unknown Role")
        org = data.get("hiringOrganization", {})
        company = org.get("name", "Unknown") if isinstance(org, dict) else str(org)
        loc_data = data.get("jobLocation", {})
        location = ""
        if isinstance(loc_data, dict):
            addr = loc_data.get("address", {})
            location = f"{addr.get('addressLocality', '')}, {addr.get('addressRegion', '')}" if isinstance(addr, dict) else str(addr)
        url = data.get("url", "")
        desc = data.get("description", "")
        if desc and BeautifulSoup:
            desc = BeautifulSoup(desc, "html.parser").get_text(strip=True)
        posted = None
        if data.get("datePosted"):
            try:
                posted = datetime.fromisoformat(data["datePosted"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        job_id = hashlib.md5(f"{source}:{title}:{company}".encode()).hexdigest()[:12]
        return JobPosting(id=job_id, title=title, company=company, location=location, url=url, source=source, description=desc[:500], posted_date=posted)

    def _deduplicate(self, jobs):
        seen = set()
        unique = []
        for job in jobs:
            key = f"{job.title.lower().strip()}|{job.company.lower().strip()}"
            if key not in seen and job.id not in self._seen_ids:
                seen.add(key)
                self._seen_ids.add(job.id)
                unique.append(job)
        return unique

    def _score_job(self, job):
        w = self.scoring
        company_lower = job.company.lower()
        for excluded in self.excluded_companies:
            if excluded in company_lower:
                return 0.0
        title_lower = job.title.lower()
        title_score = 0.0
        for target in self.target_titles:
            if target in title_lower:
                title_score = 1.0
                break
            words = target.split()
            matched = sum(1 for wd in words if wd in title_lower)
            partial = matched / len(words) if words else 0
            title_score = max(title_score, partial)
        text = f"{job.title} {job.description} {job.location}".lower()
        matched_keywords = [kw for kw in self.positive_keywords if kw in text]
        keyword_score = min(len(matched_keywords) / 5, 1.0)
        job.keywords_matched = matched_keywords
        neg_hits = sum(1 for kw in self.negative_keywords if kw in text)
        neg_penalty = min(neg_hits * w["negative_keyword_penalty"], 0.9)
        loc_lower = f"{job.location} {text}".lower()
        location_score = 1.0 if any(loc in loc_lower for loc in self.target_locations) else 0.0
        recency_score = 0.5
        if job.posted_date:
            days_old = (datetime.now() - job.posted_date.replace(tzinfo=None)).days
            recency_score = max(0, 1.0 - (days_old / 14))
        raw_score = (title_score * w["title_match_weight"] + keyword_score * w["keyword_match_weight"] + location_score * w["location_match_weight"] + recency_score * w["recency_weight"])
        return round(min(max(0.0, raw_score - neg_penalty), 1.0), 3)


class ApplicationTracker:
    def __init__(self, config=None):
        self.config = config or {}
        self.db_type = os.environ.get("ARIA_DB_TYPE", self.config.get("db_type", "mysql"))
        self._mysql_conn = None
        self._mongo_db = None
        self._initialized = False

    def _get_mysql(self):
        if pymysql is None:
            raise ImportError("pymysql required: pip install pymysql")
        if self._mysql_conn is None or not self._mysql_conn.open:
            self._mysql_conn = pymysql.connect(
                host=os.environ.get("ARIA_MYSQL_HOST", "localhost"),
                port=int(os.environ.get("ARIA_MYSQL_PORT", "3306")),
                user=os.environ.get("ARIA_MYSQL_USER", "root"),
                password=os.environ.get("ARIA_MYSQL_PASS", ""),
                database=os.environ.get("ARIA_MYSQL_DB", "aria_career"),
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True,
            )
        return self._mysql_conn

    def _get_mongo(self):
        if pymongo is None:
            raise ImportError("pymongo required: pip install pymongo")
        if self._mongo_db is None:
            uri = os.environ.get("ARIA_MONGO_URI", "mongodb://localhost:27017")
            db_name = os.environ.get("ARIA_MONGO_DB", "aria_career")
            client = pymongo.MongoClient(uri)
            self._mongo_db = client[db_name]
        return self._mongo_db

    async def initialize(self):
        if self._initialized:
            return
        loop = asyncio.get_running_loop()
        if self.db_type == "mysql":
            await loop.run_in_executor(None, self._init_mysql)
        else:
            await loop.run_in_executor(None, self._init_mongo)
        self._initialized = True
        logger.info(f"Application tracker initialized ({self.db_type})")

    def _init_mysql(self):
        conn = self._get_mysql()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS applications (
                    id VARCHAR(50) PRIMARY KEY,
                    job_id VARCHAR(50),
                    company VARCHAR(255),
                    role VARCHAR(255),
                    url TEXT,
                    status ENUM('identified','applied','interviewing','offer','rejected','withdrawn') DEFAULT 'identified',
                    applied_date DATETIME NULL,
                    follow_up_date DATETIME NULL,
                    notes TEXT,
                    source VARCHAR(100),
                    confidence_score FLOAT DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_status (status),
                    INDEX idx_company (company),
                    INDEX idx_follow_up (follow_up_date)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS job_postings (
                    id VARCHAR(50) PRIMARY KEY,
                    title VARCHAR(255),
                    company VARCHAR(255),
                    location VARCHAR(255),
                    url TEXT,
                    source VARCHAR(100),
                    description TEXT,
                    posted_date DATETIME NULL,
                    salary_range VARCHAR(100),
                    confidence FLOAT DEFAULT 0,
                    keywords_matched JSON,
                    first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_confidence (confidence),
                    INDEX idx_source (source)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

    def _init_mongo(self):
        db = self._get_mongo()
        db["applications"].create_index("status")
        db["applications"].create_index("follow_up_date")
        db["job_postings"].create_index("confidence")

    async def store_jobs(self, jobs) -> int:
        loop = asyncio.get_running_loop()
        if self.db_type == "mysql":
            return await loop.run_in_executor(None, self._store_jobs_mysql, jobs)
        else:
            return await loop.run_in_executor(None, self._store_jobs_mongo, jobs)

    def _store_jobs_mysql(self, jobs) -> int:
        conn = self._get_mysql()
        count = 0
        with conn.cursor() as cur:
            for job in jobs:
                try:
                    cur.execute("""
                        INSERT INTO job_postings (id, title, company, location, url, source, description, posted_date, salary_range, confidence, keywords_matched)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE confidence = VALUES(confidence), keywords_matched = VALUES(keywords_matched)
                    """, (job.id, job.title, job.company, job.location, job.url, job.source, job.description[:500], job.posted_date, job.salary_range, job.confidence, json.dumps(job.keywords_matched)))
                    if cur.rowcount == 1:
                        count += 1
                except Exception as e:
                    logger.debug(f"MySQL insert error: {e}")
        return count

    def _store_jobs_mongo(self, jobs) -> int:
        db = self._get_mongo()
        count = 0
        for job in jobs:
            try:
                result = db["job_postings"].update_one({"id": job.id}, {"$set": job.to_dict(), "$setOnInsert": {"first_seen": datetime.now()}}, upsert=True)
                if result.upserted_id:
                    count += 1
            except Exception as e:
                logger.debug(f"Mongo insert error: {e}")
        return count

    async def add_application(self, app) -> bool:
        loop = asyncio.get_running_loop()
        if self.db_type == "mysql":
            return await loop.run_in_executor(None, self._add_app_mysql, app)
        else:
            return await loop.run_in_executor(None, self._add_app_mongo, app)

    def _add_app_mysql(self, app) -> bool:
        conn = self._get_mysql()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO applications (id, job_id, company, role, url, status, applied_date, follow_up_date, notes, source, confidence_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE status = VALUES(status), notes = VALUES(notes), follow_up_date = VALUES(follow_up_date), updated_at = CURRENT_TIMESTAMP
            """, (app.id, app.job_id, app.company, app.role, app.url, app.status, app.applied_date, app.follow_up_date, app.notes, app.source, app.confidence_score))
        return True

    def _add_app_mongo(self, app) -> bool:
        db = self._get_mongo()
        db["applications"].update_one({"id": app.id}, {"$set": app.to_dict()}, upsert=True)
        return True

    async def get_applications(self, status=None):
        loop = asyncio.get_running_loop()
        if self.db_type == "mysql":
            return await loop.run_in_executor(None, self._get_apps_mysql, status)
        else:
            return await loop.run_in_executor(None, self._get_apps_mongo, status)

    def _get_apps_mysql(self, status):
        conn = self._get_mysql()
        with conn.cursor() as cur:
            if status:
                cur.execute("SELECT * FROM applications WHERE status = %s ORDER BY updated_at DESC", (status,))
            else:
                cur.execute("SELECT * FROM applications ORDER BY updated_at DESC")
            return cur.fetchall()

    def _get_apps_mongo(self, status):
        db = self._get_mongo()
        query = {"status": status} if status else {}
        return list(db["applications"].find(query, {"_id": 0}).sort("updated_at", -1))

    async def get_followup_due(self):
        loop = asyncio.get_running_loop()
        if self.db_type == "mysql":
            return await loop.run_in_executor(None, self._get_followup_mysql)
        else:
            return await loop.run_in_executor(None, self._get_followup_mongo)

    def _get_followup_mysql(self):
        conn = self._get_mysql()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM applications WHERE follow_up_date <= NOW() AND status IN ('applied','interviewing') ORDER BY follow_up_date ASC")
            return cur.fetchall()

    def _get_followup_mongo(self):
        db = self._get_mongo()
        return list(db["applications"].find({"follow_up_date": {"$lte": datetime.now().isoformat()}, "status": {"$in": ["applied", "interviewing"]}}, {"_id": 0}))

    async def get_pipeline_stats(self):
        loop = asyncio.get_running_loop()
        if self.db_type == "mysql":
            return await loop.run_in_executor(None, self._get_stats_mysql)
        else:
            return await loop.run_in_executor(None, self._get_stats_mongo)

    def _get_stats_mysql(self):
        conn = self._get_mysql()
        with conn.cursor() as cur:
            cur.execute("SELECT status, COUNT(*) as count FROM applications GROUP BY status")
            rows = cur.fetchall()
            cur.execute("SELECT COUNT(*) as total FROM job_postings")
            total_jobs = cur.fetchone()["total"]
            cur.execute("SELECT COUNT(*) as today_count FROM applications WHERE DATE(created_at) = CURDATE()")
            today = cur.fetchone()["today_count"]
        stats = {row["status"]: row["count"] for row in rows}
        stats["total_jobs_found"] = total_jobs
        stats["applied_today"] = today
        return stats

    def _get_stats_mongo(self):
        db = self._get_mongo()
        results = list(db["applications"].aggregate([{"$group": {"_id": "$status", "count": {"$sum": 1}}}]))
        stats = {r["_id"]: r["count"] for r in results}
        stats["total_jobs_found"] = db["job_postings"].count_documents({})
        return stats

    def close(self):
        if self._mysql_conn:
            self._mysql_conn.close()


class DailyReportGenerator:
    def __init__(self, tracker, scanner):
        self.tracker = tracker
        self.scanner = scanner

    async def generate(self) -> str:
        stats = await self.tracker.get_pipeline_stats()
        followups = await self.tracker.get_followup_due()
        recent_apps = await self.tracker.get_applications()
        today = datetime.now().strftime("%A, %B %d, %Y")
        html = f"""<html><body style="font-family:'Segoe UI',Arial,sans-serif;max-width:600px;margin:0 auto;padding:20px;">
<div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:24px;border-radius:12px 12px 0 0;">
<h1 style="color:white;margin:0;font-size:22px;">ARIA Daily Career Report</h1>
<p style="color:rgba(255,255,255,0.8);margin:4px 0 0 0;font-size:14px;">{today}</p></div>
<div style="background:white;padding:24px;border-radius:0 0 12px 12px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
<h2 style="color:#333;font-size:16px;border-bottom:2px solid #667eea;padding-bottom:8px;">Application Pipeline</h2>
<table style="width:100%;border-collapse:collapse;margin:12px 0;"><tr>
<td style="padding:8px;background:#f0f4ff;border-radius:6px;text-align:center;width:25%;"><div style="font-size:24px;font-weight:bold;color:#667eea;">{stats.get('identified',0)}</div><div style="font-size:11px;color:#666;">Identified</div></td>
<td style="padding:8px;background:#fff3e0;border-radius:6px;text-align:center;width:25%;"><div style="font-size:24px;font-weight:bold;color:#f57c00;">{stats.get('applied',0)}</div><div style="font-size:11px;color:#666;">Applied</div></td>
<td style="padding:8px;background:#e8f5e9;border-radius:6px;text-align:center;width:25%;"><div style="font-size:24px;font-weight:bold;color:#43a047;">{stats.get('interviewing',0)}</div><div style="font-size:11px;color:#666;">Interviewing</div></td>
<td style="padding:8px;background:#fce4ec;border-radius:6px;text-align:center;width:25%;"><div style="font-size:24px;font-weight:bold;color:#e53935;">{stats.get('rejected',0)}</div><div style="font-size:11px;color:#666;">Rejected</div></td>
</tr></table>
<p style="font-size:13px;color:#888;">Total jobs found: {stats.get('total_jobs_found',0)} | Applied today: {stats.get('applied_today',0)}</p>
<p style="font-size:13px;color:#43a047;">{"No follow-ups due today." if not followups else f"{len(followups)} follow-up(s) due!"}</p>
<hr style="border:none;border-top:1px solid #eee;margin:20px 0;">
<p style="font-size:11px;color:#aaa;text-align:center;">Generated by ARIA at {datetime.now().strftime('%H:%M')} | Target: 7+ applications/day</p>
</div></body></html>"""
        return html


class GmailAlertSystem:
    def __init__(self):
        self.username = os.environ.get("ARIA_EMAIL_USER", "")
        self.password = os.environ.get("ARIA_EMAIL_PASS", "")
        self.from_addr = os.environ.get("ARIA_EMAIL_FROM", self.username)
        self.to_addr = os.environ.get("ARIA_EMAIL_TO", self.username)

    @property
    def is_configured(self) -> bool:
        return bool(self.username and self.password)

    async def send_opportunity_alert(self, jobs):
        if not self.is_configured or not jobs:
            return
        subject = f"ARIA: {len(jobs)} New Opportunities Found ({datetime.now():%b %d})"
        rows = ""
        for job in jobs[:10]:
            conf_color = "#43a047" if job.confidence > 0.7 else "#f57c00" if job.confidence > 0.4 else "#666"
            rows += f'<tr><td style="padding:8px;border-bottom:1px solid #eee;"><a href="{job.url}" style="color:#667eea;font-weight:bold;">{job.title}</a><br><span style="font-size:12px;color:#666;">{job.company} - {job.location}</span></td><td style="padding:8px;border-bottom:1px solid #eee;text-align:center;"><span style="color:{conf_color};font-weight:bold;">{job.confidence*100:.0f}%</span></td><td style="padding:8px;border-bottom:1px solid #eee;font-size:12px;color:#888;">{job.source}</td></tr>'
        html = f"""<html><body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:20px;">
<div style="background:linear-gradient(135deg,#43a047,#66bb6a);padding:20px;border-radius:12px 12px 0 0;">
<h2 style="color:white;margin:0;">New Opportunities Found</h2></div>
<div style="background:white;padding:20px;border-radius:0 0 12px 12px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
<table style="width:100%;border-collapse:collapse;"><tr style="background:#fafafa;"><th style="padding:8px;text-align:left;font-size:12px;color:#666;">Role</th><th style="padding:8px;text-align:center;font-size:12px;color:#666;">Match</th><th style="padding:8px;text-align:left;font-size:12px;color:#666;">Source</th></tr>{rows}</table>
<p style="font-size:11px;color:#aaa;margin-top:16px;text-align:center;">Sent by ARIA at {datetime.now():%H:%M}</p></div></body></html>"""
        await self._send(subject, html)

    async def send_daily_report(self, report_html):
        if not self.is_configured:
            return
        await self._send(f"ARIA Daily Report - {datetime.now():%B %d, %Y}", report_html)

    async def send_followup_reminder(self, followups):
        if not self.is_configured or not followups:
            return
        items = "".join(f"<li><strong>{f.get('company','?')}</strong> - {f.get('role','?')}</li>" for f in followups)
        html = f'<html><body style="font-family:Arial,sans-serif;padding:20px;"><h2 style="color:#f57c00;">Follow-up Reminder</h2><p>{len(followups)} application(s) need follow-up:</p><ul>{items}</ul></body></html>'
        await self._send(f"ARIA: {len(followups)} Follow-ups Due Today", html)

    async def _send(self, subject, html_body):
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = self.to_addr
        msg.attach(MIMEText(html_body, "html"))
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._smtp_send, msg)
            logger.info(f"Email sent: {subject}")
        except Exception as e:
            logger.error(f"Email send failed: {e}")

    def _smtp_send(self, msg):
        import smtplib
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
