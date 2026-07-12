"""Provider-neutral web and image search client."""

import random
import requests
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..media.vlm_payload import reference_raster_bytes_are_supported
from ..signing import SearchRequest, SearchRequestSigner, apply_search_signature, load_signer

# Reject obvious non-raster URLs before GET (suffix-based; content is still validated).
_BLOCKED_REFERENCE_IMAGE_URL_SUFFIXES = (
    ".svg",
    ".svgz",
    ".pdf",
    ".eps",
    ".ps",
    ".ai",
)
_MAX_REFERENCE_IMAGE_BYTES = 25 * 1024 * 1024
# Filename extension when first SERP URL ends with one of these (must stay aligned with Pillow allowlist).
_RASTER_URL_SUFFIXES_FOR_FILENAME = (
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif",
)

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]
_BROWSER_UA = _USER_AGENTS[0]  # back-compat default


class SearchClient:
    """Client for web and image search operations"""

    def __init__(
        self,
        api_url: str | None = None,
        signer: SearchRequestSigner | None = None,
        output_dir: Optional[Path] = None,
        max_downloaded_images_per_query: Optional[int] = None
    ):
        """
        Initialize search client

        Args:
            api_url: User-configured search endpoint.
            signer: Optional user callback that authenticates each request.
            output_dir: Optional output directory for downloaded images
            max_downloaded_images_per_query: Optional cap for successful downloads per query
        """
        self.api_url = (api_url or os.environ.get("SEARCHGEN_SEARCH_API_URL") or "").strip()
        signer_spec = (os.environ.get("SEARCHGEN_SEARCH_SIGNER") or "").strip()
        self.signer = signer or (load_signer(signer_spec) if signer_spec else None)
        self.headers = {
            'Content-Type': 'application/json',
            'Connection': 'keep-alive'
        }
        self.output_dir = output_dir
        self.max_downloaded_images_per_query = max_downloaded_images_per_query

        # Persistent session with connection pooling — reuses TCP/TLS across downloads
        self._dl_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=64,
            pool_maxsize=128,
            max_retries=0,  # we handle retries at a higher level
        )
        self._dl_session.mount("https://", adapter)
        self._dl_session.mount("http://", adapter)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _post_search(self, body: Dict[str, Any]) -> requests.Response:
        if not self.api_url:
            raise RuntimeError(
                "Search is required for this row but SEARCHGEN_SEARCH_API_URL is not configured"
            )
        request = SearchRequest(
            method="POST",
            url=self.api_url,
            headers=self.headers,
            json_body=body,
        )
        signed = apply_search_signature(request, self.signer)
        return requests.post(
            signed.url,
            headers=dict(signed.headers),
            params=dict(signed.query_params),
            json=dict(signed.json_body or {}),
            timeout=30,
        )

    def _looks_mojibake(self, text: str) -> bool:
        if not isinstance(text, str) or not text:
            return False
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
        markers = ["Ã", "æ", "å", "ï¼", "ã", "â", "ð"]
        return any(marker in text for marker in markers) and not has_cjk

    def _repair_mojibake(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        if not self._looks_mojibake(text):
            return text
        try:
            repaired = text.encode("latin1").decode("utf-8")
            return repaired if repaired else text
        except Exception:
            return text

    def _decode_response_json(self, response: requests.Response) -> Dict[str, Any]:
        # Prefer UTF-8 decode from raw bytes to avoid charset mis-detection.
        raw = response.content
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return response.json()

    def _referer_for_download(self, image_url: str, page_url: Optional[str]) -> str:
        """Prefer SERP page URL as Referer; else origin of the image URL (helps avoid 403 hotlink blocks)."""
        if isinstance(page_url, str) and page_url.strip():
            pu = page_url.strip()
            if pu.startswith("http://") or pu.startswith("https://"):
                return pu
        try:
            p = urlparse(image_url)
            if p.scheme and p.netloc:
                return f"{p.scheme}://{p.netloc}/"
        except Exception:
            pass
        return ""

    def _reference_image_url_suffix_blocked(self, image_url: str) -> bool:
        try:
            path = urlparse(image_url).path.lower()
        except Exception:
            return True
        return any(path.endswith(suf) for suf in _BLOCKED_REFERENCE_IMAGE_URL_SUFFIXES)

    def _build_browser_headers(self, image_url: str, page_url: Optional[str]) -> Dict[str, str]:
        """Build realistic browser headers with UA rotation and proper sec-fetch metadata."""
        ua = random.choice(_USER_AGENTS)
        referer = self._referer_for_download(image_url, page_url)
        parsed = urlparse(image_url)
        origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""

        headers = {
            "User-Agent": ua,
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Sec-Fetch-Dest": "image",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Ch-Ua": '"Chromium";v="125", "Not=A?Brand";v="8", "Google Chrome";v="125"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"' if "Windows" in ua else '"macOS"',
            "DNT": "1",
            "Connection": "keep-alive",
        }
        if referer:
            headers["Referer"] = referer
        if origin and referer and urlparse(referer).netloc != parsed.netloc:
            headers["Origin"] = urlparse(referer).scheme + "://" + urlparse(referer).netloc

        return headers

    def _download_image(
        self,
        image_url: str,
        output_path: Path,
        timeout: int = 20,
        page_url: Optional[str] = None,
    ) -> Dict:
        """
        Download a single image from URL with browser-like behavior.

        Args:
            image_url: URL of the image to download
            output_path: Path to save the downloaded image
            timeout: Read timeout in seconds (connect timeout is always 8s)
            page_url: Optional SERP ``link`` / landing page to send as Referer

        Returns:
            Dict with success status and error message if failed
        """
        try:
            if self._reference_image_url_suffix_blocked(image_url):
                err = "URL path suffix is not an allowed reference raster format"
                self.logger.warning("Skipped download %s: %s", image_url[:160], err)
                return {"success": False, "error": err}

            headers = self._build_browser_headers(image_url, page_url)
            response = self._dl_session.get(
                image_url,
                timeout=(8, timeout),  # (connect, read) — fail fast on unreachable hosts
                stream=True,
                headers=headers,
                allow_redirects=True,
            )
            response.raise_for_status()

            chunks: List[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > _MAX_REFERENCE_IMAGE_BYTES:
                    err = f"Download exceeds max size ({_MAX_REFERENCE_IMAGE_BYTES} bytes)"
                    self.logger.warning("Failed to download %s: %s", image_url[:160], err)
                    return {"success": False, "error": err}
                chunks.append(chunk)

            raw = b"".join(chunks)
            if not reference_raster_bytes_are_supported(raw):
                err = "Body is not a supported raster image format (Pillow allowlist)"
                self.logger.warning("Rejected download %s: %s", image_url[:160], err)
                return {"success": False, "error": err}

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(raw)

            return {
                "success": True,
                "local_path": str(output_path),
                "size_bytes": len(raw),
            }

        except requests.exceptions.Timeout:
            error_msg = f"Download timeout after {timeout}s"
            self.logger.warning(f"Failed to download {image_url}: {error_msg}")
            return {"success": False, "error": error_msg}

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            self.logger.warning(f"Failed to download {image_url}: {error_msg}")
            return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"Failed to download {image_url}: {error_msg}")
            return {"success": False, "error": error_msg}

    def _normalize_title(self, title: str) -> str:
        """
        Normalize title for deduplication

        Args:
            title: Original title string

        Returns:
            Normalized title (lowercase, no extra spaces, no punctuation)
        """
        import re
        # Convert to lowercase
        normalized = title.lower()
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        # Remove common punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # Remove extra spaces again after punctuation removal
        normalized = ' '.join(normalized.split())
        return normalized

    def _deduplicate_by_title(self, image_results: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Deduplicate image results by normalized title, grouping URLs by title

        Args:
            image_results: List of image result dicts with title and imageUrl

        Returns:
            Dict mapping normalized title to list of image dicts with that title
        """
        title_groups = {}

        for img in image_results:
            title = img.get("title", "")
            if not title:
                # Use a unique key for untitled images
                normalized = f"untitled_{id(img)}"
            else:
                normalized = self._normalize_title(title)

            if normalized not in title_groups:
                title_groups[normalized] = []

            title_groups[normalized].append(img)

        self.logger.info(f"Deduplicated {len(image_results)} images into {len(title_groups)} unique titles")

        return title_groups

    def _download_first_available(
        self,
        title_group: List[Dict],
        output_path: Path,
        timeout: int = 30
    ) -> Dict:
        """
        Try downloading images from a title group, return first successful download

        Args:
            title_group: List of image dicts with same normalized title
            output_path: Path to save the downloaded image
            timeout: Download timeout per attempt

        Returns:
            Dict with success status, local_path if successful, and metadata
        """
        last_error = "No imageUrl in title group"
        attempts = 0
        for img in title_group:
            image_url = img.get("imageUrl", "")
            if not image_url:
                continue
            attempts += 1
            raw_page = (img.get("link") or img.get("source") or "")
            page_url = raw_page.strip() if isinstance(raw_page, str) and raw_page.strip() else None
            download_result = self._download_image(image_url, output_path, timeout, page_url=page_url)

            if download_result["success"]:
                # Return successful result with original image metadata
                return {
                    **img,
                    "download_success": True,
                    "local_path": download_result["local_path"],
                    "size_bytes": download_result["size_bytes"],
                    "attempted_urls": attempts,
                    "total_urls_in_group": len(title_group),
                }

            last_error = download_result.get("error") or download_result.get("download_error") or "download failed"

        # All downloads failed
        return {
            **title_group[0],  # Use first image's metadata
            "download_success": False,
            "download_error": last_error,
            "attempted_urls": attempts,
            "total_urls_in_group": len(title_group),
        }

    def _download_images_parallel(
        self,
        image_results: List[Dict],
        query_id: int = 0,
        max_workers: int = 5,
        timeout: int = 30,
        max_successful_downloads: Optional[int] = None
    ) -> List[Dict]:
        """
        Download multiple images in parallel with deduplication by title

        First deduplicates images by normalized title, then downloads the first
        available URL from each title group.

        Args:
            image_results: List of image result dicts with imageUrl
            query_id: ID of the search query (for unique filenames)
            max_workers: Maximum number of parallel downloads
            timeout: Download timeout per image
            max_successful_downloads: Stop once this many images are successfully downloaded

        Returns:
            List of image dicts with download status and local_path
        """
        if not self.output_dir:
            self.logger.warning("No output directory specified, skipping image downloads")
            return image_results

        # Create reference_images subdirectory
        images_dir = self.output_dir / "reference_images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Deduplicate by title
        title_groups = self._deduplicate_by_title(image_results)

        downloaded_results = []
        success_count = 0

        # If caller sets a strict success cap, run sequentially so we can stop early exactly at cap.
        if max_successful_downloads is not None and max_successful_downloads > 0:
            for i, (normalized_title, title_group) in enumerate(title_groups.items()):
                first_url = title_group[0].get("imageUrl", "")
                ext = ".jpg"
                if first_url and first_url.lower().endswith(_RASTER_URL_SUFFIXES_FOR_FILENAME):
                    ext = Path(first_url).suffix.lower()

                output_path = images_dir / f"q{query_id}_ref{i}{ext}"
                download_result = self._download_first_available(title_group, output_path, timeout)

                download_result["index"] = i
                download_result["query_id"] = query_id
                download_result["normalized_title"] = normalized_title
                downloaded_results.append(download_result)

                if download_result["download_success"]:
                    success_count += 1
                    self.logger.info(
                        f"Downloaded image {i}: {download_result.get('title', 'Untitled')[:50]}... "
                        f"(tried {download_result['attempted_urls']}/{download_result['total_urls_in_group']} URLs)"
                    )
                    if success_count >= max_successful_downloads:
                        self.logger.info(
                            f"Reached successful download cap for query_id={query_id}: "
                            f"{success_count}/{max_successful_downloads}. Stopping early."
                        )
                        break
                else:
                    self.logger.warning(
                        f"Failed to download image {i}: {download_result.get('download_error', 'Unknown error')} "
                        f"(tried {download_result['attempted_urls']}/{download_result['total_urls_in_group']} URLs)"
                    )

            downloaded_results.sort(key=lambda x: x["index"])
            self.logger.info(
                f"Downloaded {success_count}/{len(downloaded_results)} images successfully "
                f"(deduplicated from {len(image_results)} total results)"
            )
            return downloaded_results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download tasks for each unique title group
            future_to_title = {}
            for i, (normalized_title, title_group) in enumerate(title_groups.items()):
                # Determine file extension from first URL or use default
                first_url = title_group[0].get("imageUrl", "")
                ext = ".jpg"
                if first_url and first_url.lower().endswith(_RASTER_URL_SUFFIXES_FOR_FILENAME):
                    ext = Path(first_url).suffix.lower()

                # Use query_id in filename to avoid overwriting
                output_path = images_dir / f"q{query_id}_ref{i}{ext}"

                future = executor.submit(
                    self._download_first_available,
                    title_group,
                    output_path,
                    timeout
                )
                future_to_title[future] = (i, normalized_title, title_group)

            # Collect results
            for future in as_completed(future_to_title):
                idx, normalized_title, title_group = future_to_title[future]
                download_result = future.result()

                # Add index and query_id
                download_result["index"] = idx
                download_result["query_id"] = query_id
                download_result["normalized_title"] = normalized_title

                if download_result["download_success"]:
                    success_count += 1
                    self.logger.info(
                        f"Downloaded image {idx}: {download_result.get('title', 'Untitled')[:50]}... "
                        f"(tried {download_result['attempted_urls']}/{download_result['total_urls_in_group']} URLs)"
                    )
                else:
                    self.logger.warning(
                        f"Failed to download image {idx}: {download_result.get('download_error', 'Unknown error')} "
                        f"(tried {download_result['attempted_urls']}/{download_result['total_urls_in_group']} URLs)"
                    )

                downloaded_results.append(download_result)

        # Sort by original index
        downloaded_results.sort(key=lambda x: x["index"])

        # Log summary
        self.logger.info(
            f"Downloaded {success_count}/{len(downloaded_results)} unique images successfully "
            f"(deduplicated from {len(image_results)} total results)"
        )

        return downloaded_results

    def text_search(
        self,
        query: str,
        num: int = 10,
        country: str = "us",
        locale: str = "en",
        page: int = 1
    ) -> Dict:
        """
        Perform web/text search

        Args:
            query: Search query string
            num: Number of results to return (default 10)
            country: Country code for geolocation (default "us")
            locale: Language preference (default "en")
            page: Page number for pagination (default 1)

        Returns:
            Dict containing search results with structure:
            {
                "success": bool,
                "query": str,
                "results": List[Dict] with title, link, snippet
            }
        """
        data = {
            "search_type": "web",
            "query": query,
            "country": country,
            "locale": locale,
            "page": page,
            "num": num,
        }

        try:
            response = self._post_search(data)
            response.raise_for_status()

            result = self._decode_response_json(response)

            if result.get("success", isinstance(result.get("results"), list)):
                # Extract organic search results
                organic = result.get("results")
                if not isinstance(organic, list):
                    organic = result.get("data", {}).get("originalOutput", {}).get("organic", [])

                formatted_results = []
                for item in organic:
                    title = self._repair_mojibake(str(item.get("title", "")))
                    snippet = self._repair_mojibake(str(item.get("snippet", "")))
                    formatted_results.append({
                        "title": title,
                        "link": item.get("link", ""),
                        "snippet": snippet,
                        "position": item.get("position", 0)
                    })

                return {
                    "success": True,
                    "query": query,
                    "type": "web",
                    "results": formatted_results,
                    "total_results": len(formatted_results)
                }
            else:
                return {
                    "success": False,
                    "query": query,
                    "type": "web",
                    "error": result.get("message", "Unknown error"),
                    "results": []
                }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "query": query,
                "type": "web",
                "error": f"Request failed: {str(e)}",
                "results": []
            }
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "type": "web",
                "error": f"Unexpected error: {str(e)}",
                "results": []
            }

    def image_search(
        self,
        query: str,
        num: int = 5,  # Changed default to 5
        page: int = 1,
        download_images: bool = True,
        query_id: int = 0,
        max_downloaded_images: Optional[int] = None
    ) -> Dict:
        """
        Perform image search and optionally download images

        Args:
            query: Search query string
            num: Number of results to return (default 5)
            page: Page number for pagination (default 1)
            download_images: Whether to download images (default True)
            query_id: ID of the search query (for unique filenames, default 0)
            max_downloaded_images: Optional cap for successful downloads in this query

        Returns:
            Dict containing image search results with structure:
            {
                "success": bool,
                "query": str,
                "results": List[Dict] with title, imageUrl, local_path, etc.
            }
        """
        data = {
            "search_type": "image",
            "query": query,
            "page": page,
            "num": num,
        }

        try:
            response = self._post_search(data)
            response.raise_for_status()

            result = self._decode_response_json(response)

            if result.get("success", isinstance(result.get("results"), list)):
                # Extract image results
                images = result.get("results")
                if not isinstance(images, list):
                    images = result.get("data", {}).get("originalOutput", {}).get("images", [])

                formatted_results = []
                for item in images:
                    title = self._repair_mojibake(str(item.get("title", "")))
                    source = self._repair_mojibake(str(item.get("source", "")))
                    formatted_results.append({
                        "title": title,
                        "imageUrl": item.get("imageUrl", ""),
                        "thumbnailUrl": item.get("thumbnailUrl", ""),
                        "source": source,
                        "link": item.get("link", ""),
                        "position": item.get("position", 0)
                    })

                # Download images if requested and output_dir is set
                if download_images and self.output_dir:
                    self.logger.info(f"Downloading {len(formatted_results)} images for query: {query}")
                    effective_cap = (
                        max_downloaded_images
                        if max_downloaded_images is not None
                        else self.max_downloaded_images_per_query
                    )
                    formatted_results = self._download_images_parallel(
                        formatted_results,
                        query_id=query_id,
                        max_successful_downloads=effective_cap
                    )

                return {
                    "success": True,
                    "query": query,
                    "type": "image",
                    "results": formatted_results,
                    "total_results": len(formatted_results)
                }
            else:
                return {
                    "success": False,
                    "query": query,
                    "type": "image",
                    "error": result.get("message", "Unknown error"),
                    "results": []
                }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "query": query,
                "type": "image",
                "error": f"Request failed: {str(e)}",
                "results": []
            }
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "type": "image",
                "error": f"Unexpected error: {str(e)}",
                "results": []
            }

    def search(
        self,
        query: str,
        search_type: str = "web",
        num: int = 10,
        query_id: int = 0,
        max_downloaded_images: Optional[int] = None
    ) -> Dict:
        """
        Unified search interface

        Args:
            query: Search query string
            search_type: "web" or "image"
            num: Number of results to return
            query_id: ID of the search query (for unique filenames)
            max_downloaded_images: Optional cap for successful image downloads per query

        Returns:
            Search results dict
        """
        if search_type == "image":
            return self.image_search(
                query,
                num=num,
                query_id=query_id,
                max_downloaded_images=max_downloaded_images
            )
        else:
            return self.text_search(query, num=num)
