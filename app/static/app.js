const fileInput = document.getElementById('fileInput');
const recordBtn = document.getElementById('recordBtn');
const responseEl = document.getElementById('response');

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file && file.type.startsWith('audio')) {
    sendAudio(file);
  }
});

recordBtn.addEventListener('click', async () => {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        sendAudio(audioBlob);
      };

      mediaRecorder.start();
      isRecording = true;

      // Change to red while recording
      recordBtn.classList.remove('bg-gray-400', 'hover:bg-gray-500');
      recordBtn.classList.add('bg-red-600', 'hover:bg-red-700');
    } catch (err) {
      console.error('Microphone access denied:', err);
      responseEl.textContent = 'Microphone access denied.';
    }
  } else {
    mediaRecorder.stop();
    isRecording = false;

    // Reset to gray when stopped
    recordBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
    recordBtn.classList.add('bg-gray-400', 'hover:bg-gray-500');
  }
});

function sendAudio(audioBlob) {
  const formData = new FormData();
  formData.append('file', audioBlob, 'recording.webm');  // ✅ Fix here

  responseEl.textContent = 'Analyzing...';

  fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
  })
    .then((res) => res.json())
    .then((data) => {
      responseEl.textContent = `Detected Emotion: ${data.emotion || 'Success'}`;
    })
    .catch((err) => {
      console.error(err);
      responseEl.textContent = 'Error sending audio to backend.';
    });
}
