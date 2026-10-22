# Nurture Chromium prototype

1. Open `chrome://extensions`, enable Developer mode and choose **Load unpacked** → this `extension/` directory.
2. Open an ordinary HTTP(S) page. Click the extension icon, enter your English word rules, check authorization and select **Apply to this tab**.
3. Each replacement is followed by a small `*`. Use the on-page **Stop & restore** button or the popup’s stop button. Reloading clears this document’s session; new documents require another explicit activation.

Only `activeTab` and `scripting` permissions are requested. There are no host permissions, remote scripts, stored settings, telemetry or network requests. Existing sessions can remain active on more than one explicitly enabled tab. Popup closure does not stop them. Original matched words are held only in that page session’s memory for restoration; other page scripts can see DOM changes.

Ordinary text only: no media, OCR, audio, frames, shadow DOM, code blocks or editable controls. Matches do not span separate DOM nodes. Case-insensitive whole-word matching uses the same engine as the site. It is not a trained classifier or a guarantee of complete filtering. Dynamic sites can overwrite changes or react unexpectedly. Stop/restore does not overwrite a replacement span that the site has subsequently changed; reload if necessary.

The engine is copied from `docs/filter.js`; keep the files byte-identical. No build step or GitHub Actions is required.
