# Uploads

Use `upload_file(...)` for real file inputs first, and only fall back to coordinate clicks when the site hides the input behind custom UI. Include both local upload flow and remote-browser flow here, because remote sessions need explicit file availability and cannot assume access to the operator's local disk.
