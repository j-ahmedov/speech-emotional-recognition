<h2>Speech Emotion Recognition</h2>

This project is a full project with ready trained AI model to detect the emotion of the speaker. You can either upload an audio file or record your voice.<br><br>
Before running this project make sure you have installed the followings on your computer:<br>
<ul>
<li><a href="https://www.ffmpeg.org/download.html">ffmpeg</a> package</li>
<li><a href="https://www.python.org/">python</a> (recommended versions: 3.10-3.13)</li>
</ul>
<br>

Then you should open terminal on your project and run following to install necessary libraries:
<pre>pip3 install -r requirements.txt</pre>
<br>

Then you should type this command to run the project:
<pre>uvicorn app.main:app --host 0.0.0.0 --port 8000</pre>
