(() => {
  'use strict';

  const byId = (id) => document.getElementById(id);
  const localCapture = document.documentElement.dataset.captureMode === 'local';
  const ui = {
    camera: byId('camera-input'),
    gallery: byId('gallery-input'),
    captureButton: byId('capture-button'),
    galleryButton: byId('gallery-button'),
    received: byId('received-count'),
    maximum: byId('maximum-count'),
    progress: byId('progress-fill'),
    connection: byId('connection-state'),
    connectionText: byId('connection-text'),
    status: byId('status-box'),
    statusText: byId('status-text'),
    retry: byId('retry-button'),
    expiry: byId('expiry-text'),
    recent: byId('recent-photos'),
    recentSection: byId('recent-section'),
    download: byId('download-button'),
    atelier: byId('atelier-link'),
    scanStart: byId('scan-start'),
    scanStop: byId('scan-stop'),
    scanPanel: byId('live-scan'),
    scanVideo: byId('scan-video'),
    scanFeedback: byId('scan-feedback'),
    scanExplanation: byId('scan-explanation'),
    scanCount: byId('scan-count'),
  };
  const params = new URLSearchParams(location.hash.slice(1));
  const session = params.get('session') || '';
  let key = params.get('key') || '';
  const valid =
    localCapture ||
    (/^[A-Za-z0-9_-]{12,128}$/.test(session) && /^[A-Za-z0-9_-]{24,256}$/.test(key));
  // Capabilities stay in memory, never in query strings, storage, or referrers.
  try {
    history.replaceState(null, '', location.pathname);
  } catch {
    /* Fragment is never sent in HTTP. */
  }
  params.delete('key');
  params.delete('session');

  let received = 0,
    maximum = 32,
    connected = false,
    stopped = !valid;
  let busy = false,
    disposed = false,
    polling = false,
    pollTimer = 0;
  const activeRequests = new Set();
  const previews = [];
  const localPhotos = [];
  let localBytes = 0;
  let scanning = false,
    cameraStarting = false,
    cameraGeneration = 0;
  let cameraStream = null,
    scanTimer = 0,
    previousFrame = null,
    acceptedFrame = null;
  let lastCapture = 0;
  const scanAvailable = isSecureContext && !!navigator.mediaDevices?.getUserMedia;

  class SessionError extends Error {}
  const message = (text, error = false) => {
    ui.statusText.textContent = text;
    ui.status.classList.toggle('error', error);
  };
  const connection = (text, state = '') => {
    ui.connectionText.textContent = text;
    ui.connection.className = `connection ${state}`;
  };
  const controls = () => {
    ui.scanCount.textContent = `${received} / ${maximum} vues`;
    const disabled = busy || !connected || stopped || disposed || received >= maximum;
    ui.captureButton.disabled = disabled || scanning || cameraStarting;
    ui.galleryButton.disabled = disabled || scanning || cameraStarting;
    ui.camera.disabled = disabled || scanning || cameraStarting;
    ui.gallery.disabled = disabled || scanning || cameraStarting;
    ui.scanStart.disabled = disabled || scanning || cameraStarting || !scanAvailable;
    ui.scanStart.hidden = scanning;
    if (ui.atelier) ui.atelier.tabIndex = scanning || cameraStarting ? -1 : 0;
    ui.scanStop.disabled = false;
    ui.scanStart.textContent = cameraStarting
      ? 'Autorisez la caméra…'
      : 'Scanner la pièce en direct';
    ui.retry.disabled = busy || polling;
    ui.captureButton.setAttribute('aria-busy', String(busy));
    if (ui.download)
      ui.download.disabled = busy || scanning || cameraStarting || !localPhotos.length;
  };
  const stopSession = (text) => {
    stopScan();
    stopped = true;
    connected = false;
    key = '';
    clearTimeout(pollTimer);
    connection('Session terminée', 'problem');
    message(text, true);
    ui.retry.hidden = true;
    controls();
  };

  function stopScan(text = '') {
    cameraGeneration++;
    scanning = false;
    cameraStarting = false;
    clearTimeout(scanTimer);
    for (const track of cameraStream?.getTracks() || []) track.stop();
    cameraStream = null;
    ui.scanVideo.pause();
    ui.scanVideo.srcObject = null;
    ui.scanPanel.hidden = true;
    previousFrame = acceptedFrame = null;
    controls();
    if (!document.hidden && !disposed && scanAvailable) ui.scanStart.focus();
    if (text) message(text);
  }

  function frameDifference(a, b) {
    let difference = 0;
    // Remove a global brightness shift before comparing the image contents.
    let offset = 0;
    for (let i = 0; i < a.length; i++) offset += a[i] - b[i];
    offset /= a.length;
    for (let i = 0; i < a.length; i++) difference += Math.abs(a[i] - b[i] - offset);
    return difference / a.length;
  }

  function sampleFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = 160;
    canvas.height = 120;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('Ce navigateur ne peut pas lire les vues de la caméra.');
    context.drawImage(ui.scanVideo, 0, 0, 160, 120);
    const pixels = context.getImageData(0, 0, 160, 120).data;
    const gray = new Float32Array(160 * 120);
    let brightness = 0,
      edges = 0;
    for (let i = 0; i < gray.length; i++) {
      gray[i] = pixels[i * 4] * 0.299 + pixels[i * 4 + 1] * 0.587 + pixels[i * 4 + 2] * 0.114;
      brightness += gray[i];
    }
    for (let y = 1; y < 119; y++)
      for (let x = 1; x < 159; x++) {
        const i = y * 160 + x;
        const laplacian = gray[i - 1] + gray[i + 1] + gray[i - 160] + gray[i + 160] - 4 * gray[i];
        edges += laplacian * laplacian;
      }
    return { gray, brightness: brightness / gray.length, detail: edges / (158 * 118) };
  }

  async function scanTick(generation) {
    if (!scanning || generation !== cameraGeneration || disposed || stopped) return;
    try {
      if (!busy && ui.scanVideo.readyState >= 2) {
        const frame = sampleFrame();
        const motion = previousFrame ? frameDifference(frame.gray, previousFrame) : Infinity;
        previousFrame = frame.gray;
        const changed = !acceptedFrame || frameDifference(frame.gray, acceptedFrame) >= 6;
        if (frame.brightness < 25)
          ui.scanFeedback.textContent = 'La pièce est trop sombre. Ajoutez de la lumière.';
        else if (frame.detail < 12)
          ui.scanFeedback.textContent =
            'Visez des surfaces avec des détails et laissez la mise au point se faire.';
        else if (motion > 24)
          ui.scanFeedback.textContent = 'Ralentissez un instant pour garder une vue nette.';
        else if (!changed)
          ui.scanFeedback.textContent =
            'Déplacez-vous un peu pour montrer un nouveau point de vue.';
        else if (performance.now() - lastCapture >= 1800) {
          const canvas = document.createElement('canvas');
          const ratio = Math.min(
            1,
            2048 / Math.max(ui.scanVideo.videoWidth, ui.scanVideo.videoHeight),
          );
          canvas.width = Math.max(1, Math.round(ui.scanVideo.videoWidth * ratio));
          canvas.height = Math.max(1, Math.round(ui.scanVideo.videoHeight * ratio));
          const context = canvas.getContext('2d');
          if (!context) throw new Error('La vue ne peut pas être préparée.');
          context.drawImage(ui.scanVideo, 0, 0, canvas.width, canvas.height);
          const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
          canvas.width = canvas.height = 1;
          if (!scanning || generation !== cameraGeneration) return;
          if (!blob) throw new Error('La vue ne peut pas être enregistrée.');
          const before = received;
          await sendFiles([new File([blob], `Scan-${Date.now()}.jpg`, { type: 'image/jpeg' })]);
          if (!scanning || generation !== cameraGeneration) return;
          if (received === before) {
            stopScan(); // Preserve the actionable error from sendFiles.
            return;
          }
          acceptedFrame = frame.gray;
          lastCapture = performance.now();
          ui.scanFeedback.textContent = `Vue ${received} enregistrée. Continuez lentement : angles, murs, sol et plafond.`;
          if (received >= maximum) {
            stopScan(
              localCapture
                ? '32 vues enregistrées. La caméra est fermée. Téléchargez le ZIP pour poursuivre sur le PC.'
                : '32 vues transmises. La caméra est fermée. Poursuivez dans l’atelier sur le PC.',
            );
            return;
          }
          if (!connected) {
            stopScan();
            return;
          }
        }
      }
    } catch (error) {
      if (generation !== cameraGeneration) return;
      stopScan();
      message(
        error.message || 'La caméra a été interrompue. Vos vues déjà enregistrées sont conservées.',
        true,
      );
    }
    if (scanning && generation === cameraGeneration)
      scanTimer = setTimeout(() => void scanTick(generation), 450);
  }

  async function startScan() {
    if (
      busy ||
      stopped ||
      disposed ||
      scanning ||
      cameraStarting ||
      !connected ||
      received >= maximum
    )
      return;
    if (!scanAvailable) {
      message(
        'Le scan en direct demande HTTPS et un navigateur compatible. Utilisez le site public dans Safari ou Chrome, ou prenez des photos une par une.',
        true,
      );
      return;
    }
    const generation = ++cameraGeneration;
    cameraStarting = true;
    ui.scanPanel.hidden = false;
    ui.scanFeedback.textContent =
      'Autorisez la caméra arrière pour démarrer. Aucun son n’est enregistré.';
    controls();
    ui.scanStop.focus();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
      });
      if (generation !== cameraGeneration || disposed || stopped || document.hidden) {
        for (const track of stream.getTracks()) track.stop();
        if (generation === cameraGeneration) stopScan();
        return;
      }
      cameraStream = stream;
      for (const track of stream.getVideoTracks())
        track.addEventListener(
          'ended',
          () => {
            if (generation === cameraGeneration)
              stopScan(
                'La caméra a été interrompue. Vos vues sont conservées ; vous pouvez reprendre le scan.',
              );
          },
          { once: true },
        );
      ui.scanVideo.srcObject = stream;
      await ui.scanVideo.play();
      if (generation !== cameraGeneration) return;
      cameraStarting = false;
      scanning = true;
      previousFrame = acceptedFrame = null;
      lastCapture = -Infinity;
      ui.scanFeedback.textContent =
        'Avancez lentement. Les vues sont enregistrées automatiquement.';
      controls();
      void scanTick(generation);
    } catch (error) {
      if (generation !== cameraGeneration) return;
      stopScan();
      const text =
        error.name === 'NotAllowedError'
          ? 'L’accès à la caméra a été refusé. Autorisez-la dans les réglages du site, puis réessayez. Vous pouvez aussi prendre des photos une par une.'
          : error.name === 'NotFoundError'
            ? 'Aucune caméra disponible. Essayez sur votre téléphone ou importez des photos.'
            : 'La caméra ne peut pas démarrer. Fermez les autres applications qui l’utilisent, puis réessayez ou prenez des photos une par une.';
      message(text, true);
    }
  }

  async function request(path, options = {}, timeoutMs = 12000) {
    if (stopped || disposed) throw new SessionError('La session n’est plus active.');
    const controller = new AbortController();
    activeRequests.add(controller);
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const headers = new Headers(options.headers);
      headers.set('Authorization', `Bearer ${key}`);
      const response = await fetch(path, {
        ...options,
        headers,
        signal: controller.signal,
        credentials: 'omit',
        cache: 'no-store',
        referrerPolicy: 'no-referrer',
      });
      if (response.status === 403 || response.status === 410 || response.status === 404) {
        const text =
          'Cette session a expiré ou a été fermée sur l’ordinateur. Ouvrez un nouveau lien depuis RoomCraft sur le PC.';
        stopSession(text);
        throw new SessionError(text);
      }
      if (!response.ok) {
        let detail = '';
        try {
          const body = await response.json();
          if (typeof body.detail === 'string') detail = body.detail;
        } catch {
          /* Keep HTTP status. */
        }
        throw new Error(detail || `L’ordinateur a refusé la requête (${response.status}).`);
      }
      const result = await response.json();
      if (stopped || disposed) throw new SessionError('La session n’est plus active.');
      return result;
    } finally {
      clearTimeout(timeout);
      activeRequests.delete(controller);
    }
  }

  function schedulePoll() {
    clearTimeout(pollTimer);
    if (!localCapture && !stopped && !disposed && !document.hidden)
      pollTimer = setTimeout(() => {
        void refresh();
      }, 5000);
  }

  async function refresh(manual = false) {
    if (localCapture) {
      connected = true;
      received = localPhotos.length;
      ui.received.textContent = String(received);
      ui.maximum.textContent = `/ ${maximum}`;
      ui.progress.style.width = `${(received / maximum) * 100}%`;
      connection('Photos conservées sur cet appareil', 'connected');
      controls();
      return true;
    }
    if (stopped || disposed || busy || polling) {
      schedulePoll();
      return false;
    }
    polling = true;
    controls();
    try {
      const status = await request(`/api/sessions/${encodeURIComponent(session)}`);
      if (
        !Number.isInteger(status.received_count) ||
        !Number.isInteger(status.max_photos) ||
        status.max_photos < 1 ||
        status.max_photos > 32 ||
        status.received_count < 0 ||
        status.received_count > status.max_photos
      )
        throw new Error('Le compteur renvoyé par l’ordinateur est illisible.');
      received = status.received_count;
      maximum = status.max_photos;
      connected = true;
      ui.received.textContent = String(received);
      ui.maximum.textContent = `/ ${maximum}`;
      ui.progress.style.width = `${(100 * received) / maximum}%`;
      connection('Votre ordinateur est connecté', 'connected');
      const expiry =
        typeof status.expires_at === 'number'
          ? new Date(status.expires_at * 1000)
          : new Date(status.expires_at);
      ui.expiry.textContent = Number.isNaN(expiry.getTime())
        ? ''
        : `Lien temporaire · fin prévue à ${expiry.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}. Pour rouvrir la page, utilisez le lien ou le QR code du PC.`;
      if (received >= maximum)
        message(
          'Votre série est complète. Retrouvez les photos dans RoomCraft sur l’ordinateur pour vérifier les vues et lancer la reconstruction.',
        );
      else if (manual || ui.statusText.textContent === 'Préparation de votre session…')
        message(
          received
            ? 'La connexion est rétablie. Vérifiez les photos reçues sur le PC avant de renvoyer une prise dont l’envoi n’était pas confirmé.'
            : 'Prêt pour la première photo. Commencez par un mur, puis déplacez-vous progressivement autour de la pièce.',
        );
      ui.retry.hidden = true;
      return true;
    } catch (error) {
      if (stopped || disposed || error instanceof SessionError) return false;
      connected = false;
      connection('Ordinateur temporairement injoignable', 'problem');
      if (!busy)
        message(
          'La connexion a été interrompue. Gardez RoomCraft ouvert sur le PC et vérifiez que les deux appareils sont sur le même Wi-Fi.',
          true,
        );
      ui.retry.hidden = false;
      return false;
    } finally {
      polling = false;
      controls();
      schedulePoll();
    }
  }

  async function imageDimensions(file) {
    const bytes = new Uint8Array(await file.slice(0, 1_048_576).arrayBuffer());
    const view = new DataView(bytes.buffer);
    if (bytes.length >= 24 && view.getUint32(0) === 0x89504e47 && view.getUint32(4) === 0x0d0a1a0a)
      return [view.getUint32(16), view.getUint32(20)];
    if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8) {
      let offset = 2;
      while (offset + 4 < bytes.length) {
        if (bytes[offset] !== 0xff) break;
        while (bytes[offset] === 0xff) offset++;
        const marker = bytes[offset++];
        if (marker === 0xd9 || marker === 0xda) break;
        if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
        if (offset + 2 > bytes.length) break;
        const length = view.getUint16(offset);
        if (length < 2) break;
        if (
          [0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf].includes(
            marker,
          ) &&
          offset + 7 <= bytes.length
        )
          return [view.getUint16(offset + 5), view.getUint16(offset + 3)];
        offset += length;
      }
      throw new Error(
        'Les dimensions de cette photo JPEG ne peuvent pas être lues. Exportez-la à nouveau en JPEG ou PNG.',
      );
    }
    if (
      bytes.length >= 30 &&
      String.fromCharCode(...bytes.slice(0, 4)) === 'RIFF' &&
      String.fromCharCode(...bytes.slice(8, 12)) === 'WEBP'
    ) {
      const kind = String.fromCharCode(...bytes.slice(12, 16));
      if (kind === 'VP8X')
        return [
          1 + bytes[24] + (bytes[25] << 8) + (bytes[26] << 16),
          1 + bytes[27] + (bytes[28] << 8) + (bytes[29] << 16),
        ];
      if (kind === 'VP8 ' && bytes[23] === 0x9d && bytes[24] === 0x01 && bytes[25] === 0x2a)
        return [view.getUint16(26, true) & 0x3fff, view.getUint16(28, true) & 0x3fff];
      if (kind === 'VP8L' && bytes[20] === 0x2f)
        return [
          1 + bytes[21] + ((bytes[22] & 0x3f) << 8),
          1 + (bytes[22] >> 6) + (bytes[23] << 2) + ((bytes[24] & 0x0f) << 10),
        ];
    }
    if (
      file.type === 'image/gif' ||
      file.type === 'image/svg+xml' ||
      /\.(gif|svg)$/i.test(file.name)
    )
      throw new Error(
        'Choisissez une photographie JPEG, PNG ou WebP. Les animations et dessins vectoriels ne sont pas acceptés.',
      );
    // HEIC/HEIF decoding is device-dependent. Do not pretend to convert a
    // format the browser cannot read; the image load below reports it clearly.
    return null;
  }

  async function prepare(file) {
    if (!file.size || file.size > 50_000_000)
      throw new Error(
        'Une photo est vide ou dépasse 50 Mo. Choisissez une version moins volumineuse.',
      );
    const dimensions = await imageDimensions(file);
    if (
      dimensions &&
      (!dimensions[0] || !dimensions[1] || dimensions[0] * dimensions[1] > 80_000_000)
    )
      throw new Error(
        'Une photo dépasse 80 mégapixels. Réduisez sa résolution avant de l’envoyer.',
      );
    const url = URL.createObjectURL(file);
    const image = new Image();
    let timer;
    try {
      await new Promise((resolve, reject) => {
        timer = setTimeout(
          () =>
            reject(
              new Error(
                'La lecture de cette photo prend trop de temps. Choisissez une version JPEG plus petite.',
              ),
            ),
          20000,
        );
        image.onload = resolve;
        image.onerror = () =>
          reject(
            new Error(
              'Cette photo ne peut pas être lue sur ce téléphone. Pour HEIC/HEIF, choisissez une version JPEG ou prenez une nouvelle photo dans un format compatible.',
            ),
          );
        image.src = url;
      });
      if (stopped || disposed) throw new SessionError('La session n’est plus active.');
      const width = image.naturalWidth,
        height = image.naturalHeight;
      if (!width || !height || width * height > 80_000_000)
        throw new Error('Les dimensions de cette photo sont invalides ou dépassent 80 mégapixels.');
      // Current browsers apply EXIF orientation when displaying/drawing the
      // image. The output JPEG has pixels in that orientation, without EXIF.
      const ratio = Math.min(1, 2048 / Math.max(width, height));
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(width * ratio));
      canvas.height = Math.max(1, Math.round(height * ratio));
      try {
        const context = canvas.getContext('2d');
        if (!context)
          throw new Error(
            'Ce navigateur ne peut pas préparer la photo. Essayez Safari ou Chrome à jour.',
          );
        context.fillStyle = '#ffffff';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
        if (!blob || blob.type !== 'image/jpeg' || blob.size > 10_000_000)
          throw new Error(
            'La photo ne peut pas être préparée en JPEG de moins de 10 Mo. Choisissez une version plus petite.',
          );
        const stem =
          (file.name || 'photo')
            .replace(/\.[^.]+$/, '')
            .replace(/[^\p{L}\p{N} _-]/gu, '-')
            .slice(0, 80) || 'photo';
        return { blob, name: `${stem}.jpg` };
      } finally {
        canvas.width = 1;
        canvas.height = 1;
      }
    } finally {
      clearTimeout(timer);
      image.onload = null;
      image.onerror = null;
      image.src = '';
      URL.revokeObjectURL(url);
    }
  }

  function showSent(photo, name) {
    const item = document.createElement('li');
    const img = new Image();
    img.alt = '';
    img.width = 56;
    img.height = 43;
    img.decoding = 'async';
    const url = URL.createObjectURL(photo.blob);
    img.src = url;
    const detail = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = name;
    const info = document.createElement('small');
    info.textContent = `${Math.max(1, Math.round(photo.blob.size / 1000))} Ko · ${localCapture ? 'sur cet appareil' : 'reçu sur l’ordinateur'}`;
    const check = document.createElement('span');
    check.className = 'sent-check';
    check.textContent = '✓';
    check.setAttribute('aria-hidden', 'true');
    detail.append(title, info);
    item.append(img, detail, check);
    ui.recent.prepend(item);
    previews.push({ url, item });
    while (previews.length > 6) {
      const previous = previews.shift();
      previous.item.remove();
      URL.revokeObjectURL(previous.url);
    }
    ui.recentSection.hidden = false;
  }

  async function sendFiles(files) {
    if (busy || stopped || disposed || !connected || !files.length) return;
    if (files.length > maximum - received) {
      message(
        `Il reste ${maximum - received} emplacement(s) dans cette série. Choisissez moins de photos.`,
        true,
      );
      return;
    }
    busy = true;
    controls();
    clearTimeout(pollTimer);
    let confirmed = 0;
    try {
      for (let i = 0; i < files.length; i++) {
        message(`Préparation de la photo ${i + 1} sur ${files.length}…`);
        const photo = await prepare(files[i]);
        if (stopped || disposed) throw new SessionError('La session n’est plus active.');
        if (localCapture) {
          if (localBytes + photo.blob.size > 80_000_000)
            throw new Error('La série dépasse 80 Mo. Téléchargez-la avant de continuer.');
          localPhotos.push(photo);
          localBytes += photo.blob.size;
          confirmed++;
          showSent(photo, `Photo ${String(localPhotos.length).padStart(2, '0')}`);
          await refresh();
          continue;
        }
        const form = new FormData();
        form.append('file', photo.blob, photo.name);
        message(`Envoi de la photo ${i + 1} sur ${files.length}… Gardez cette page ouverte.`);
        let result;
        try {
          result = await request(
            `/api/sessions/${encodeURIComponent(session)}/photos`,
            { method: 'POST', body: form },
            30000,
          );
        } catch (error) {
          if (error instanceof SessionError) throw error;
          connected = false;
          connection('Envoi à vérifier sur le PC', 'problem');
          ui.retry.hidden = false;
          throw new Error(
            `${error.message || 'Connexion interrompue.'} L’envoi n’est pas confirmé et ne sera pas réessayé automatiquement. Vérifiez les photos reçues sur le PC avant de renvoyer cette prise de vue.`,
          );
        }
        if (!result || typeof result.id !== 'string')
          throw new Error(
            'La confirmation reçue est illisible. Vérifiez les photos sur le PC avant de renvoyer cette prise.',
          );
        confirmed++;
        showSent(photo, typeof result.name === 'string' ? result.name : photo.name);
        // Read the authoritative counter; never increment it optimistically.
        const status = await request(`/api/sessions/${encodeURIComponent(session)}`);
        if (
          !Number.isInteger(status.received_count) ||
          !Number.isInteger(status.max_photos) ||
          status.max_photos < 1 ||
          status.max_photos > 32 ||
          status.received_count < 0 ||
          status.received_count > status.max_photos
        )
          throw new Error(
            'Le compteur est illisible. Vérifiez la série sur le PC avant de continuer.',
          );
        received = status.received_count;
        maximum = status.max_photos;
        ui.received.textContent = String(received);
        ui.maximum.textContent = `/ ${maximum}`;
        ui.progress.style.width = `${(100 * received) / maximum}%`;
      }
      if (localCapture)
        message(
          `${confirmed} photo${confirmed > 1 ? 's' : ''} ajoutée${confirmed > 1 ? 's' : ''}. Téléchargez le ZIP avant de fermer cette page. Aucune image n’est envoyée à un serveur.`,
        );
      else
        message(
          received >= maximum
            ? 'Votre série est complète. Vous pouvez poursuivre la reconstruction et la calibration dans RoomCraft sur l’ordinateur.'
            : `${confirmed === 1 ? 'Photo reçue' : `${confirmed} photos reçues`} sur le PC. Déplacez-vous un peu avant la prochaine prise de vue.`,
        );
    } catch (error) {
      if (!stopped && !disposed && !(error instanceof SessionError)) {
        message(error.message || 'La photo n’a pas pu être préparée ou envoyée.', true);
        ui.retry.hidden = false;
      }
    } finally {
      busy = false;
      controls();
      schedulePoll();
    }
  }

  ui.captureButton.addEventListener('click', () => {
    ui.camera.value = '';
    ui.camera.click();
  });
  ui.scanStart.addEventListener('click', () => void startScan());
  ui.scanPanel.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') ui.scanStop.click();
    if (event.key === 'Tab') {
      event.preventDefault();
      ui.scanStop.focus();
    }
  });
  ui.scanStop.addEventListener('click', () =>
    stopScan(
      localCapture
        ? 'Scan arrêté, caméra fermée. Vos vues sont conservées. Téléchargez le ZIP pour poursuivre la reconstruction sur le PC.'
        : 'Scan arrêté, caméra fermée. Vos vues transmises restent dans l’atelier sur le PC.',
    ),
  );
  if (!scanAvailable)
    ui.scanExplanation.textContent =
      'Le scan en direct nécessite HTTPS et un navigateur compatible. Ouvrez le site public dans Safari ou Chrome. Ici, utilisez « Prendre une photo » ou la galerie.';
  ui.galleryButton.addEventListener('click', () => {
    ui.gallery.value = '';
    ui.gallery.click();
  });
  for (const input of [ui.camera, ui.gallery])
    input.addEventListener('change', () => {
      const files = Array.from(input.files || []);
      input.value = '';
      void sendFiles(files);
    });
  ui.retry.addEventListener('click', () => {
    void refresh(true);
  });
  ui.download?.addEventListener('click', async () => {
    if (!localCapture || busy || !localPhotos.length) return;
    busy = true;
    controls();
    message('Préparation de votre archive photo…');
    try {
      const { zipSync, strToU8 } = await import('./zip.mjs');
      const files = {};
      const manifest = {
        format: 'roomcraft-photo-set',
        version: 1,
        createdAt: new Date().toISOString(),
        photos: [],
      };
      for (let i = 0; i < localPhotos.length; i++) {
        const name = `Photo-${String(i + 1).padStart(2, '0')}.jpg`;
        files[name] = new Uint8Array(await localPhotos[i].blob.arrayBuffer());
        manifest.photos.push({ name, bytes: localPhotos[i].blob.size });
      }
      files['roomcraft-photos.json'] = strToU8(JSON.stringify(manifest, null, 2));
      files['LIRE-MOI.txt'] = strToU8(
        'RoomCraft — photos uniquement, sans dimensions déduites. Décompressez ce ZIP sur le PC, puis importez les JPEG dans « À partir de photos ». La reconstruction multivue locale reste expérimentale et exige une mesure réelle.',
      );
      const zip = zipSync(files, { level: 0 });
      const url = URL.createObjectURL(new Blob([zip], { type: 'application/zip' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = `RoomCraft-photos-${new Date().toISOString().slice(0, 10)}.zip`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 30000);
      message(
        'Téléchargement lancé. Transférez le ZIP sur votre PC, décompressez-le et importez les JPEG dans RoomCraft. Vos photos restent aussi sur cette page jusqu’à sa fermeture.',
      );
    } catch (error) {
      message(
        error.message || 'Le ZIP n’a pas pu être créé. Gardez cette page ouverte et réessayez.',
        true,
      );
    } finally {
      busy = false;
      controls();
    }
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      clearTimeout(pollTimer);
      if (scanning || cameraStarting)
        stopScan(
          'Scan en pause : caméra fermée car vous avez quitté la page. Vos vues sont conservées. Appuyez sur « Scanner la pièce en direct » pour reprendre.',
        );
    } else if (!busy) void refresh();
  });
  addEventListener('pagehide', () => {
    stopScan();
    disposed = true;
    clearTimeout(pollTimer);
    for (const controller of activeRequests) controller.abort();
    for (const preview of previews.splice(0)) {
      preview.item.remove();
      URL.revokeObjectURL(preview.url);
    }
  });
  addEventListener('pageshow', (event) => {
    if (event.persisted && !stopped) {
      disposed = false;
      void refresh();
    }
  });
  if (localCapture) {
    ui.download.hidden = false;
    ui.atelier.hidden = false;
    byId('page-title').innerHTML = 'Faites le tour<br>de votre pièce.';
    document.querySelector('.intro > p:last-child').textContent =
      'Ouvrez la caméra et déplacez-vous lentement : RoomCraft prend les vues automatiquement. La reconstruction à partir de ces images se poursuit sur votre PC.';
    document.querySelector('.brand p').textContent = 'Votre pièce, depuis votre téléphone';
    document.querySelector('.count small').textContent = 'sur cet appareil';
    byId('recent-title').textContent = 'Dernières photos ajoutées';
    ui.expiry.textContent =
      'Les photos restent uniquement dans cette page. Téléchargez le ZIP avant de la fermer ou de la recharger.';
    const footer = document.querySelector('footer');
    footer.replaceChildren();
    for (const text of [
      'Aucune photo n’est envoyée à GitHub ou à un serveur. Le ZIP est créé directement sur votre téléphone.',
      'Pour reconstruire : transférez le ZIP sur le PC, décompressez-le, ouvrez RoomCraft puis importez les JPEG dans « À partir de photos ».',
      'GitHub Pages héberge cette interface ; la reconstruction Python reste sur le PC et nécessite une mesure réelle. Ces prises de vue ne constituent pas un relevé exact.',
    ]) {
      const p = document.createElement('p');
      p.textContent = text;
      footer.append(p);
    }
    message(
      'Prêt pour la première photo. Visez des vues nettes qui se recouvrent, en vous déplaçant.',
    );
    void refresh();
  } else if (!valid) {
    key = '';
    connection('Lien de prise de vues manquant', 'problem');
    message(
      'Ouvrez « Scanner avec mon téléphone » dans RoomCraft sur l’ordinateur, puis utilisez le lien ou le QR code proposé. Les deux appareils doivent être sur le même Wi-Fi.',
      true,
    );
    controls();
  } else {
    void refresh();
  }
})();
