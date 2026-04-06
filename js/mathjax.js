window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

function typesetMath() {
  if (!window.MathJax || typeof window.MathJax.typesetPromise !== "function") {
    return;
  }

  if (typeof window.MathJax.typesetClear === "function") {
    window.MathJax.typesetClear();
  }

  window.MathJax.typesetPromise();
}

window.addEventListener("load", typesetMath);

if (typeof document$ !== "undefined" && typeof document$.subscribe === "function") {
  document$.subscribe(() => {
    typesetMath();
  });
}
