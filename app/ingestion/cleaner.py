from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString, Tag

_REMOVE_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "form",
        "nav",
        "aside",
        "button",
        "img",
    }
)
_NOISE_CLASS_RE = re.compile(
    r"(advert|adsense|cta|share-|social|newsletter|popup|related-posts|wp-block-embed)",
    re.I,
)
_SHORTCODE_RE = re.compile(r"\[[a-zA-Z][^\[\]]{0,400}\]")
_JUMP_TO_RE = re.compile(r"jump\s+to", re.I)
_NBSP_RE = re.compile(r"(?:\xa0|&nbsp;)+", re.I)


def clean_html(html: str) -> str:
    """Strip WordPress presentation noise and return cleaned HTML."""
    if not html or not html.strip():
        return ""

    soup = BeautifulSoup(html, "lxml")
    _decompose_noise(soup)
    _remove_jump_to_toc(soup)
    _strip_shortcodes(soup)
    _normalize_nbsp(soup)
    _remove_empty_nodes(soup)
    return _fragment(soup)


def _decompose_noise(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(_REMOVE_TAGS):
        tag.decompose()
    for tag in soup.find_all(class_=_NOISE_CLASS_RE):
        tag.decompose()


def _remove_jump_to_toc(soup: BeautifulSoup) -> None:
    for text in list(soup.find_all(string=_JUMP_TO_RE)):
        start = text.parent if isinstance(text.parent, Tag) else None
        if start is None:
            continue
        sibling = start.find_next_sibling()
        if isinstance(sibling, Tag) and sibling.name == "ul" and _is_anchor_list(sibling):
            sibling.decompose()
        start.decompose()


def _is_anchor_list(ul: Tag) -> bool:
    links = ul.find_all("a", href=True)
    if not links:
        return False
    return all(href.startswith("#") for href in (a.get("href", "") for a in links))


def _strip_shortcodes(soup: BeautifulSoup) -> None:
    for text in list(soup.find_all(string=_SHORTCODE_RE)):
        cleaned = _SHORTCODE_RE.sub("", str(text))
        text.replace_with(cleaned)


def _normalize_nbsp(soup: BeautifulSoup) -> None:
    for text in list(soup.find_all(string=_NBSP_RE)):
        cleaned = _NBSP_RE.sub(" ", str(text))
        text.replace_with(cleaned)


def _remove_empty_nodes(soup: BeautifulSoup) -> None:
    for tag in list(soup.find_all(["p", "div", "span", "li"])):
        if tag.get_text(strip=True):
            continue
        if tag.find(["img", "table", "ul", "ol"]):
            continue
        tag.decompose()


def _fragment(soup: BeautifulSoup) -> str:
    body = soup.body
    if body is None:
        return str(soup)
    return "".join(
        str(child) for child in body.children if not isinstance(child, NavigableString) or str(child).strip()
    )
