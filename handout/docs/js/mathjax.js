window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: false,
    processEnvironments: false
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

function typesetMath(retries = 20) {
  if (!window.MathJax || typeof window.MathJax.typesetPromise !== "function") {
    if (retries > 0) {
      window.setTimeout(() => typesetMath(retries - 1), 150);
    }
    return;
  }

  const run = () => window.MathJax.typesetPromise();

  if (window.MathJax.startup?.promise) {
    window.MathJax.startup.promise.then(run);
    return;
  }

  run();
}

document$.subscribe(() => {
  typesetMath();
});

window.addEventListener("load", () => {
  typesetMath();
});
