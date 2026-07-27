// Layout Script
// Injects the shared header and footer dynamically using values from data.js
// Highlights the current page in the navigation bar.

// Clean URL: Redirect /index.html to /
if (window.location.pathname.endsWith('/index.html')) {
  const cleanPath = window.location.pathname.replace(/\/index\.html$/, '/') + window.location.search + window.location.hash;
  window.location.replace(cleanPath);
}

document.addEventListener("DOMContentLoaded", () => {
  // Check if PORTFOLIO_DATA exists
  if (!window.PORTFOLIO_DATA) {
    console.error("data.js not loaded or PORTFOLIO_DATA is missing.");
    return;
  }

  const profile = window.PORTFOLIO_DATA.profile;

  // 1. Injects common Head metadata and links dynamically (for pages where it hasn't been set statically)
  const docTitle = document.title;
  if (!docTitle.includes(profile.name)) {
    document.title = `${docTitle} — ${profile.name}`;
  }

  // 2. Inject Header
  const headerEl = document.querySelector("header");
  if (headerEl) {
    headerEl.className = "mb-14";
    headerEl.innerHTML = `
      <a href="./" class="logo">${profile.name}</a>
      <nav class="font-sans mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[13px] text-muted-foreground">
        <a href="./" class="nav-link" id="nav-about">About</a>
        <a href="research.html" class="nav-link" id="nav-research">Research</a>
        <a href="projects.html" class="nav-link" id="nav-projects">Projects</a>
        <a href="publications.html" class="nav-link" id="nav-publications">Publications</a>
        <a href="blog.html" class="nav-link" id="nav-blog">Blog</a>
        <a href="cv.html" class="nav-link" id="nav-cv">CV</a>
      </nav>
    `;
  }

  // 3. Inject Footer
  const footerEl = document.querySelector("footer");
  if (footerEl) {
    footerEl.className = "font-sans mt-24 pt-6 rule text-[12px] text-muted-foreground flex flex-wrap gap-x-4 gap-y-1";
    footerEl.innerHTML = `
      <a href="mailto:${profile.email}">Email</a>
      <a href="${profile.github}" target="_blank" rel="noreferrer">GitHub</a>
      <a href="${profile.linkedin}" target="_blank" rel="noreferrer">LinkedIn</a>
      <a href="${profile.twitter}" target="_blank" rel="noreferrer">X</a>
      <span class="ml-auto"> Last updated ${profile.lastUpdated}</span>
    `;
  }

  // 4. Highlight active nav item
  const currentPath = window.location.pathname;
  const filename = currentPath.substring(currentPath.lastIndexOf('/') + 1) || 'index.html';

  let activeId = "";
  if (filename === 'index.html' || filename === '') {
    activeId = "nav-about";
  } else if (filename === 'research.html') {
    activeId = "nav-research";
  } else if (filename === 'projects.html') {
    activeId = "nav-projects";
  } else if (filename === 'publications.html') {
    activeId = "nav-publications";
  } else if (filename === 'blog.html' || filename === 'blog-post.html') {
    activeId = "nav-blog";
  } else if (filename === 'cv.html') {
    activeId = "nav-cv";
  }

  if (activeId) {
    const activeLink = document.getElementById(activeId);
    if (activeLink) {
      activeLink.classList.add("active");
    }
  }
});
