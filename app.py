from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from auto_mixer import AutoMixer
import threading
import logging
import time
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['OUTPUT_FOLDER'] = 'output'

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables to track mixing status
mixing_status = {
    'is_mixing': False,
    'progress': 0,
    'result': None,
    'error': None,
    'status_message': None
}

ALLOWED_EXTENSIONS = {'mp3', 'wav'}
FFMPEG_PATH = "C:\\Users\\mitta\\Downloads\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe"
FFPROBE_PATH = "C:\\Users\\mitta\\Downloads\\ffmpeg-master-latest-win64-gpl\\bin\\ffprobe.exe"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_old_files():
    """Clean up old files from uploads and output directories"""
    current_time = time.time()
    max_age = 3600  # 1 hour

    for directory in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                if current_time - os.path.getmtime(filepath) > max_age:
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        logger.error(f"Error removing old file {filepath}: {str(e)}")

def update_mixing_progress(stage, progress):
    """Update mixing progress with stage information"""
    global mixing_status
    mixing_status['progress'] = progress
    mixing_status['status_message'] = stage

def mix_songs_task(song1_path, song2_path):
    """Background task to mix songs"""
    global mixing_status
    try:
        # Initialize AutoMixer with FFmpeg paths
        update_mixing_progress("Initializing mixer...", 10)
        mixer = AutoMixer(FFMPEG_PATH, FFPROBE_PATH)
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"mix_{timestamp}.mp3"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Mix the songs
        update_mixing_progress("Analyzing songs...", 20)
        update_mixing_progress("Detecting chorus sections...", 30)
        update_mixing_progress("Finding best transitions...", 50)
        update_mixing_progress("Creating final mix...", 70)
        
        mixes = mixer.mix_songs(song1_path, song2_path, output_path)
        
        if mixes:
            update_mixing_progress("Mix completed successfully!", 100)
            mixing_status['result'] = output_filename
        else:
            mixing_status['error'] = "No suitable mixes were created"
            
    except Exception as e:
        logger.error(f"Error mixing songs: {str(e)}")
        mixing_status['error'] = str(e)
    finally:
        # Clean up uploaded files
        try:
            os.remove(song1_path)
            os.remove(song2_path)
        except Exception as e:
            logger.error(f"Error cleaning up uploaded files: {str(e)}")
        
        mixing_status['is_mixing'] = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    global mixing_status
    
    if mixing_status['is_mixing']:
        return jsonify({'error': 'Another mixing process is already running'}), 400
    
    # Reset mixing status
    mixing_status = {
        'is_mixing': True,
        'progress': 0,
        'result': None,
        'error': None,
        'status_message': 'Starting upload...'
    }
    
    try:
        if 'song1' not in request.files or 'song2' not in request.files:
            return jsonify({'error': 'Both songs must be provided'}), 400
        
        song1 = request.files['song1']
        song2 = request.files['song2']
        
        if song1.filename == '' or song2.filename == '':
            return jsonify({'error': 'Both songs must be selected'}), 400
        
        if not (allowed_file(song1.filename) and allowed_file(song2.filename)):
            return jsonify({'error': 'Invalid file format. Only MP3 and WAV files are allowed'}), 400
        
        # Clean old files
        clean_old_files()
        
        # Save uploaded files with original names (sanitized)
        song1_filename = secure_filename(song1.filename)
        song2_filename = secure_filename(song2.filename)
        
        song1_path = os.path.join(app.config['UPLOAD_FOLDER'], song1_filename)
        song2_path = os.path.join(app.config['UPLOAD_FOLDER'], song2_filename)
        
        song1.save(song1_path)
        song2.save(song2_path)
        
        # Start mixing process in background
        mixing_thread = threading.Thread(
            target=mix_songs_task,
            args=(song1_path, song2_path)
        )
        mixing_thread.daemon = True  # Make thread daemon so it doesn't block app shutdown
        mixing_thread.start()
        
        return jsonify({'message': 'Mixing process started'}), 200
        
    except Exception as e:
        mixing_status['is_mixing'] = False
        mixing_status['error'] = str(e)
        logger.error(f"Error in upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def get_status():
    return jsonify(mixing_status)

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {filename}")
            
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True) 