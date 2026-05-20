from __future__ import annotations
import time
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any
from pathlib import Path
from urllib.parse import urlparse
import requests
from .config import CrawlerConfig
from .policies import CrawlPolicies
from .scraper import extract_document , extract_links
from .storage import CrawlStorage

class WebCrawler:
    def __init__(self, config : CrawlerConfig):
        self.config = config
        self.policies = CrawlPolicies(config)
        self.storage = CrawlStorage(config.output_dir)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.user_agent})

        self.last_request_at : dict[str,float] = {}
        self.queued : set[str] = set()
        self.visited_urls : set[str] = set()
        if self.config.persist_visited:
            self.visited_urls = self.load_visited_urls(self.config.visited_urls_path)

        self.stats : dict[str , int] = {
            "seed_urls" : len(config.seed_urls),
            "urls_visited" : 0,
            "pages_fetched" : 0,
            "documents_saved" : 0,
            "links_discovered" : 0,
            "links_enqueued" : 0,
            "errors" : 0,
            "skipped_policy" : 0,
            "skipped_robots" : 0,
            "skipped_non_html" : 0,
            "skipped_duplicate" : 0,
            "skipped_persisted" : 0,
        }
    
    def respect_delay(self , url : str) -> None:
        host = urlparse(url).netloc.lower()
        now = time.monotonic()
        last = self.last_request_at.get(host)
        if last is not None:
            sleep_time = self.config.request_delay - (now - last)
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.last_request_at[host] = time.monotonic()
    
    def record_error(self , url : str , depth : int , reason : str , detail : str | None = None) -> None:
        self.stats["errors"] += 1
        self.storage.append_error(
            {
                "url" : url,
                "depth" : depth,
                "reason" : reason,
                "detail" : detail,
                "timestamp" : datetime.now(UTC).isoformat()
            }
        )
    
    def fetch(self , url : str , depth : int) -> tuple[str,str,str] | None:
        self.respect_delay(url)
        try:
            response = self.session.get(url , timeout=self.config.timeout , allow_redirects=True)
        except requests.RequestException as exc:
            self.record_error(url , depth , "request-exception", str(exc))
            return None
        
        final_url = self.policies.normalize_url(url , response.url) or response.url
        if response.status_code >= 400:
            self.record_error(
                final_url,
                depth,
                "http_error",
                f"status_code={response.status_code}"
            )
            return None
        
        if not self.policies.is_allowed(final_url):
            self.stats["skipped_policy"] += 1
            return None
        
        response.encoding = response.apparent_encoding or response.encoding
        content_type = response.headers.get("Content-Type","").lower()
        return final_url , response.text , content_type
    
    def crawl(self) -> dict[str,Any]:
        start_time = datetime.now(UTC)
        queue: deque[tuple[str,int,str | None]] = deque()
        visited : set[str] = set()

        for seed in self.config.seed_urls:
            normalized = self.policies.normalize_url(seed , seed)
            if not normalized:
                continue
            if not self.policies.is_allowed(normalized):
                self.stats["skipped_policy"] += 1
                continue
            queue.append((normalized,0,None))
            self.queued.add(normalized)
        
        while queue and self.stats["pages_fetched"] < self.config.max_pages:
            current_url , depth , parent_url = queue.popleft()
            self.queued.discard(current_url)

            if current_url in visited:
                self.stats["skipped_duplicate"] += 1
                continue

            if not self.policies.is_allowed(current_url):
                self.stats["skipped_policy"] += 1
                continue

            if not self.policies.is_allowed_by_robots(current_url,self.config.user_agent):
                self.stats["skipped_robots"] += 1
                continue

            if self.config.persist_visited and current_url in self.visited_urls:
                visited.add(current_url)
                self.stats["urls_visited"] += 1
                self.stats["skipped_persisted"] += 1

                fetched = self.fetch(current_url, depth)
                if fetched is None:
                    continue
                final_url, html, content_type = fetched
                if "text/html" not in content_type:
                    self.stats["skipped_non_html"] += 1
                    continue
                self.stats["pages_fetched"] += 1

                links = extract_links(html, final_url, self.policies)
                self.stats["links_discovered"] += len(links)
                for link in links:
                    if link in visited or link in self.queued:
                        continue
                    queue.append((link, depth + 1, final_url))
                    self.queued.add(link)
                    self.stats["links_enqueued"] += 1

                if self.config.persist_visited:
                    self.append_visited_url(final_url)

                self.print_progress()
                continue

            visited.add(current_url)
            self.stats["urls_visited"] += 1

            fetched = self.fetch(current_url,depth)
            if fetched is None:
                continue
            final_url,html,content_type = fetched

            if "text/html" not in content_type:
                self.stats["skipped_non_html"] += 1
                continue

            self.stats["pages_fetched"] += 1
            document = extract_document(html,final_url)
            document["depth"] = depth
            document["parent_url"] = parent_url

            raw_html_path = None
            if self.config.save_html:
                raw_html_path = self.storage.save_html(final_url,html,document["doc_id"])
                document["raw_html_path"] = raw_html_path
            
            self.storage.append_document(document)
            self.stats["documents_saved"] += 1
            self.print_progress()
            if self.config.persist_visited:
                self.append_visited_url(final_url)

            if depth >= self.config.max_depth:
                continue

            links = extract_links(html,final_url,self.policies)
            self.stats["links_discovered"] += len(links)
            for link in links:
                if link in visited or link in self.queued:
                    continue
                queue.append((link,depth+1,final_url))
                self.queued.add(link)
                self.stats["links_enqueued"] += 1
        
        end_time = datetime.now(UTC)
        elapsed_seconds = (end_time - start_time).total_seconds()
        report : dict[str,Any] = {
            "run_id" : self.storage.run_id,
            "started_at" : start_time.isoformat(),
            "finished_at" : end_time.isoformat(),
            "elapsed_seconds" : round(elapsed_seconds,3),
            "config" : {
                **asdict(self.config),
                "output_dir" : str(self.config.output_dir),
                "allowed_domains" : sorted(self.config.allowed_domains),
            },
            "stats" : self.stats,
            "paths" : {
                "html_dir" : str(self.storage.html_dir),
                "documents_jsonl" : str(self.storage.documents_path),
                "errors_jsonl" : str(self.storage.errors_path),
            },
        }
        report_path = self.storage.save_report(report)
        report["paths"]["report_json"] = report_path
        return report

    def print_progress(self) -> None:
        pages = self.stats["pages_fetched"]
        if pages <= 0:
            return

        by_pages = (
            self.config.progress_every_pages > 0
            and pages % self.config.progress_every_pages == 0
        )
        if not by_pages:
            return
        print(
            f"[crawler] pages={pages} saved={self.stats['documents_saved']} "
            f"queued={len(self.queued)} visited={self.stats['urls_visited']}"
        )

    def load_visited_urls(self, path: Path) -> set[str]:
        urls: set[str] = set()
        try:
            if not path.exists():
                return urls
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    url = line.strip()
                    if url:
                        urls.add(url)
        except OSError:
            return urls
        return urls

    def append_visited_url(self, url: str) -> None:
        if not url or url in self.visited_urls:
            return
        self.visited_urls.add(url)
        path = self.config.visited_urls_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(url)
                file.write("\n")
        except OSError:
            return