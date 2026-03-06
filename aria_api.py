"""
ARIA Dashboard API Server
Connects the browser dashboard to your MongoDB data.
Run: python aria_api.py
Then open http://localhost:5100 in your browser.
"""

import os
import json
import asyncio
import hashlib
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

try:
    import pymongo
    MONGO_URI = os.environ.get("ARIA_MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB = os.environ.get("ARIA_MONGO_DB", "aria_career")
    client = pymongo.MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    print(f"Connected to MongoDB: {MONGO_DB}")
except ImportError:
    print("ERROR: pymongo required"); exit(1)
except Exception as e:
    print(f"ERROR: MongoDB connection failed: {e}"); exit(1)

try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from aria_integrations import OpportunityScanner, ApplicationTracker
    scanner = OpportunityScanner()
    tracker = ApplicationTracker({"db_type": "mongodb"})
    SCANNER_AVAILABLE = True
    print("Scanner loaded")
except ImportError as e:
    SCANNER_AVAILABLE = False
    print(f"Scanner not available: {e}")

# ── Resume Generator integration ──────────────────────────────────────────────
RESUME_GEN_PATH = os.path.expanduser("~/Documents/memory-brain-integration")
try:
    sys.path.insert(0, RESUME_GEN_PATH)
    from archetype_resume_generator import ResumeGenerator
    resume_generator = ResumeGenerator()
    RESUME_GEN_AVAILABLE = True
    print("Resume generator loaded")
except ImportError as e:
    RESUME_GEN_AVAILABLE = False
    print(f"Resume generator not available: {e}")

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime): return obj.isoformat()
        if hasattr(obj, '__str__'): return str(obj)
        return super().default(obj)

class APIHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200); self._cors(); self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        params = parse_qs(urlparse(self.path).query)
        if path == "/api/jobs": self._get_jobs(params)
        elif path == "/api/applications": self._get_applications(params)
        elif path == "/api/stats": self._get_stats()
        elif path == "/api/health": self._json({"status":"ok","scanner":SCANNER_AVAILABLE,"resume_gen":RESUME_GEN_AVAILABLE})
        elif path == "/" or path.endswith(".html"): self._serve_dashboard()
        else: self._json({"error":"Not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path; body = self._body()
        if path == "/api/scan": self._trigger_scan()
        elif path == "/api/jobs": self._add_job(body)
        elif path == "/api/applications": self._add_application(body)
        elif path == "/api/generate-resume": self._generate_resume(body)
        else: self._json({"error":"Not found"}, 404)

    def do_PUT(self):
        path = urlparse(self.path).path; body = self._body()
        if path.startswith("/api/applications/"): self._update_app(path.split("/")[-1], body)
        elif path.startswith("/api/jobs/"): self._update_job(path.split("/")[-1], body)
        else: self._json({"error":"Not found"}, 404)

    def _get_jobs(self, params):
        query = {}
        if "source" in params: query["source"] = params["source"][0]
        jobs = list(db["job_postings"].find(query, {"_id":0}).sort("confidence",-1).limit(200))
        app_map = {a["job_id"]:a["status"] for a in db["applications"].find({},{"_id":0,"job_id":1,"status":1}) if "job_id" in a}
        for j in jobs:
            j["status"] = app_map.get(j.get("id",""), "new")
            j.setdefault("keywords_matched",[]); j.setdefault("confidence",0)
            j.setdefault("location",""); j.setdefault("company","Unknown")
        self._json(jobs)

    def _get_applications(self, params):
        query = {}
        if "status" in params: query["status"] = params["status"][0]
        self._json(list(db["applications"].find(query,{"_id":0}).sort("updated_at",-1)))

    def _get_stats(self):
        pipe = list(db["applications"].aggregate([{"$group":{"_id":"$status","count":{"$sum":1}}}]))
        stats = {r["_id"]:r["count"] for r in pipe}
        stats["total_jobs"] = db["job_postings"].count_documents({})
        self._json(stats)

    def _trigger_scan(self):
        if not SCANNER_AVAILABLE: self._json({"error":"Scanner not available"},503); return
        def run():
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
            try:
                async def do_scan():
                    await tracker.initialize()
                    jobs = await scanner.scan_all()
                    if jobs: stored = await tracker.store_jobs(jobs); print(f"Scan: {len(jobs)} found, {stored} new")
                loop.run_until_complete(do_scan())
            finally: loop.close()
        threading.Thread(target=run, daemon=True).start()
        self._json({"status":"scan_started","message":"Refresh in ~30 seconds"})

    def _add_job(self, body):
        if not body.get("title"): self._json({"error":"title required"},400); return
        jid = body.get("id") or hashlib.md5(f"manual:{body['title']}:{body.get('company','')}".encode()).hexdigest()[:12]
        doc = {"id":jid,"title":body.get("title",""),"company":body.get("company",""),"location":body.get("location","TBD"),
               "url":body.get("url",""),"source":body.get("source","Manual"),"description":"",
               "posted_date":datetime.now().isoformat(),"confidence":float(body.get("confidence",0.5)),
               "keywords_matched":body.get("keywords_matched",[]),"first_seen":datetime.now()}
        db["job_postings"].update_one({"id":jid},{"$set":doc},upsert=True)
        self._json({"status":"created","id":jid},201)

    def _add_application(self, body):
        jid = body.get("job_id") or body.get("id")
        if not jid: self._json({"error":"job_id required"},400); return
        doc = {"id":jid,"job_id":jid,"company":body.get("company",""),"role":body.get("role",body.get("title","")),
               "url":body.get("url",""),"status":body.get("status","identified"),
               "applied_date":body.get("applied_date"),"follow_up_date":body.get("follow_up_date"),
               "notes":body.get("notes",""),"source":body.get("source",""),
               "confidence_score":float(body.get("confidence",0)),
               "created_at":datetime.now().isoformat(),"updated_at":datetime.now().isoformat()}
        db["applications"].update_one({"job_id":jid},{"$set":doc},upsert=True)
        self._json({"status":"created"},201)

    def _update_app(self, jid, body):
        upd = {"updated_at":datetime.now().isoformat()}
        if "status" in body:
            upd["status"] = body["status"]
            if body["status"]=="applied": upd.setdefault("applied_date",datetime.now().isoformat())
        if "notes" in body: upd["notes"]=body["notes"]
        job = db["job_postings"].find_one({"id":jid},{"_id":0})
        if job:
            upd.setdefault("company",job.get("company","")); upd.setdefault("role",job.get("title",""))
            upd.setdefault("url",job.get("url","")); upd.setdefault("source",job.get("source",""))
            upd.setdefault("job_id",jid); upd.setdefault("id",jid)
        db["applications"].update_one({"$or":[{"job_id":jid},{"id":jid}]},{"$set":upd},upsert=True)
        self._json({"status":"updated"})

    def _update_job(self, jid, body):
        db["job_postings"].update_one({"id":jid},{"$set":{k:v for k,v in body.items() if k!="id"}})
        self._json({"status":"updated"})

    def _generate_resume(self, body):
        if not RESUME_GEN_AVAILABLE:
            self._json({"error": "Resume generator not available — check ~/Documents/memory-brain-integration"}, 503)
            return
        company  = body.get("company", "").strip()
        role     = body.get("role", "").strip()
        score    = float(body.get("score", 0.75))
        job_id   = body.get("job_id") or f"{company.lower().replace(' ','_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if not company or not role:
            self._json({"error": "company and role are required"}, 400)
            return
        try:
            # Optionally write JD to memory brain DB so archetype classifier uses it
            jd_text = body.get("jd", "")
            if jd_text:
                import sqlite3
                db_path = os.path.expanduser("~/Documents/memory-brain/central_memory.db")
                try:
                    conn = sqlite3.connect(db_path, timeout=10.0)
                    conn.execute("""
                        INSERT OR REPLACE INTO jobs_master
                        (job_id, company, role, description, semantic_score, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (job_id, company, role, jd_text, score, datetime.now().isoformat()))
                    conn.commit(); conn.close()
                except Exception: pass  # DB write optional — generator still runs

            path = resume_generator.generate_resume(
                job_id=job_id, company=company, role=role, score=score
            )
            self._json({
                "status": "generated",
                "path": str(path),
                "filename": os.path.basename(str(path)),
                "job_id": job_id,
                "company": company,
                "role": role,
            })
        except Exception as e:
            self._json({"error": f"Generation failed: {str(e)}"}, 500)

    def _serve_dashboard(self):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),"aria_dashboard.html")
        try:
            with open(p) as f: content=f.read()
            self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(content.encode())
        except FileNotFoundError: self._json({"error":"aria_dashboard.html not found"},404)

    def _body(self):
        n = int(self.headers.get("Content-Length",0))
        if n==0: return {}
        try: return json.loads(self.rfile.read(n))
        except: return {}

    def _json(self, data, status=200):
        self.send_response(status); self._cors()
        self.send_header("Content-Type","application/json"); self.end_headers()
        self.wfile.write(json.dumps(data,cls=JSONEncoder).encode())

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET,POST,PUT,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")

    def log_message(self, fmt, *args): print(f"[API] {args[0]}")

def main():
    port = int(os.environ.get("ARIA_API_PORT","5100"))
    server = HTTPServer(("0.0.0.0",port), APIHandler)
    print(f"\n  ARIA Dashboard API — http://localhost:{port}\n  Scanner: {'Ready' if SCANNER_AVAILABLE else 'Not loaded'}\n  Ctrl+C to stop\n")
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nStopped."); server.server_close()

if __name__ == "__main__": main()
