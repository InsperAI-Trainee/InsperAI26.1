function loadEmbed(button, url) {
  const wrapper = button.parentElement;
  const iframe = document.createElement('iframe');
  iframe.src = url;
  iframe.title = button.textContent.trim();
  iframe.loading = 'eager';
  wrapper.replaceChild(iframe, button);
}