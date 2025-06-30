from dotenv import load_dotenv
import os
import re
import requests
from flask import Flask, render_template, request, send_from_directory
from openai import OpenAI
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import cv2
import numpy as np
import uuid
import argparse
from werkzeug.utils import secure_filename
import subprocess
import json
import io


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecret")

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

UPLOAD_FOLDER = "static/uploads"
THUMBNAIL_FOLDER = "static/thumbnails"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["THUMBNAIL_FOLDER"] = THUMBNAIL_FOLDER

# ========== Utility Functions ==========

def get_channel_id(channel_name):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": channel_name,
        "type": "channel",
        "maxResults": 1,
        "key": YOUTUBE_API_KEY
    }
    res = requests.get(url, params=params).json()
    if "items" not in res or not res["items"]:
        raise ValueError(f"No channel found for: {channel_name}\n\nFull response: {res}")
    return res["items"][0]["snippet"]["channelId"]

def get_similar_titles(keywords):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        "part": "snippet",
        "q": keywords,
        "type": "video",
        "maxResults": 10,
        "key": YOUTUBE_API_KEY
    }
    search_response = requests.get(search_url, params=search_params).json()
    video_items = search_response.get("items", [])
    video_ids = [item["id"]["videoId"] for item in video_items]

    if not video_ids:
        return []

    video_url = "https://www.googleapis.com/youtube/v3/videos"
    video_params = {
        "part": "snippet,statistics",
        "id": ",".join(video_ids),
        "key": YOUTUBE_API_KEY
    }
    video_response = requests.get(video_url, params=video_params).json()
    videos = video_response.get("items", [])

    high_performers = []
    for vid in videos:
        stats = vid.get("statistics", {})
        snippet = vid.get("snippet", {})
        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        published = snippet.get("publishedAt", "")
        title = snippet.get("title", "")

        ratio = likes / views if views else 0
        pub_date = datetime.strptime(published[:10], "%Y-%m-%d")
        age_months = (datetime.now() - pub_date).days / 30

        if views >= 10000 and ratio >= 0.02 and age_months <= 12:
            high_performers.append(f"{title} (Views: {views}, Like Ratio: {round(ratio*100,1)}%)")

    return high_performers

def improve_description_prompt(channel_name, video_title, video_desc):
    prompt = (
        f"You are a YouTube SEO expert. The channel is '{channel_name}'. The video title is '{video_title}'. "
        f"Expand and enrich the following video description with more relevant keywords, emotional appeal, and context. "
        "Make it significantly longer, more detailed, and better for SEO. Start your result directly with the improved description (no labels):\n\n"
        f"{video_desc}"
    )
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return res.choices[0].message.content.strip()

def generate_questions(channel_name, video_title, video_desc):
    prompt = (
        f"You are helping a YouTube SEO expert optimize a new video. "
        f"The channel name is '{channel_name}'. The video title is '{video_title}'. The current description is:\n{video_desc}\n"
        "Generate 3-5 thoughtful, concise questions that will help clarify the video's target audience, content, unique value, and anything else that would improve YouTube SEO. "
        "Do not answer the questions. Return as a list."
    )
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    content = res.choices[0].message.content.strip()
    questions = [q.strip("-1234567890. ").strip() for q in content.split('\n') if q.strip() and "?" in q]
    return questions[:5]

def optimize_title_and_desc(channel_name, video_title, video_desc, similar_titles, user_answers):
    summary_table = """
Description Length Guide:
<300 characters: Vlogs, comedy, entertainment (Low/Medium SEO)
300–1000: Gaming, mainstream, music (Medium SEO)
1000–2000: Tutorials, competitive niches (High SEO)
2000–5000: Deep dives, info-rich (Highest SEO)
"""
    context = f"""
Channel: {channel_name}
Video Title: {video_title}
User-Improved Description: {video_desc}
Similar Top-Performing Titles: {similar_titles}
Extra User Context: {user_answers}
"""

    brand_voice_note = ""
    lower_name = channel_name.lower()
    
    if "abu garcia" in lower_name:
        brand_voice_note = (
            "\nThis is an Abu Garcia-branded channel. Reflect their voice by channeling the mindset of high-performance anglers: confident, competitive, and always pushing to improve. Think of the tone as focused and driven, but down-to-earth — preparation, precision, and the will to win matter. Highlight themes like:\n"
            "- Fishing to win is a mindset\n"
            "- Preparation breeds confidence\n"
            "- Every detail matters\n"
            "Include words like: Confident, Bold, Win, Relentless, Training, Champion, Focused, Driven."
        )
    elif "berkley" in lower_name:
        brand_voice_note = (
            "\nThis is a Berkley-branded channel. Keep the tone light, smart, and fun — like someone who’s serious about catching fish but doesn’t take themselves too seriously. Mix playfulness with expertise. Highlight themes like:\n"
            "- Talk smarter, not harder\n"
            "- Have fun, try something new\n"
            "- Science = catching more fish\n"
            "Good words to include: Science, Chemistry, Optimized, Formulated, Proven, Performance Enhancer."
        )
    elif "penn" in lower_name:
        brand_voice_note = (
            "\nThis is a PENN-branded channel. Speak with earned authority and confidence. Think of the tone like a seasoned captain — direct, no fluff, battle-ready. Highlight grit, strength, and respect for the fight. Core themes include:\n"
            "- Speak with authority\n"
            "- Prepare for battle\n"
            "- Endure and triumph\n"
            "Include words like: Battle, Ready, Strength, Durable, Tough, Trust, Adrenaline, Tried & True."
        )
    elif "fenwick" in lower_name:
    brand_voice_note = (
        "\nThis is a Fenwick-branded channel. Speak with quiet confidence, full-body awareness, and an eye for craftsmanship. Fenwick's voice is self-assured without being loud—it speaks through experience, not hype. Highlight themes like:\n"
        "- Doing over talking\n"
        "- Fully engaged senses\n"
        "- Craftsmanship in every detail\n"
        "Include words like: Ultimate sensitivity, Intentional, Methodical, Precision, Expert design, Feel the action, Technology meets technique, Palm fit, Finesse, Composed, Distinctive."
    )
    elif "frabill" in lower_name:
    brand_voice_note = (
        "\nThis is a Frabill-branded channel. The tone is focused, direct, and deeply rooted in trust and performance. Frabill messaging is solution-driven, designed to motivate and support serious anglers who expect reliability in the harshest conditions. Highlight themes like:\n"
        "- Leaning forward with innovation and grit\n"
        "- Fishing success built on trusted performance\n"
        "- Speaking plainly but powerfully to the serious angler\n"
        "Good words to include: Focused, Trusted, Principled, Direct, Certain, Inspiring, Rigid control, Innovative, Performance-driven, Community, Proficiency, Progress."
    )
    elif "plano" in lower_name:
    brand_voice_note = (
        "\nThis is a Plano-branded channel. The voice is dependable, no-nonsense, and rooted in preparedness. Plano content should reflect a clear, organized approach to outdoor readiness — focusing on smart storage, intentional gear setup, and trust in long-lasting performance. Highlight themes like:\n"
        "- Be Plano Ready: Organized gear, prepared mindset\n"
        "- Purpose-built design for real-world adventure\n"
        "- Confidence through protection and durability\n"
        "Good words to include: Ready, Prepared, Gear, Intentional, Purpose-built, Organized, Compartment, Practical, Protect, Access, Adventure."
    )
    elif "shakespeare" in lower_name:
    brand_voice_note = (
        "\nThis is a Shakespeare-branded channel. The tone is relaxed, approachable, and full of joy. Shakespeare is all about simple pleasures and getting outside with friends and family. Speak in short, bite-sized sentences. Keep copy easy to follow, fun, and filled with a sense of anticipation and wonder. Highlight themes like:\n"
        "- Keep it simple, light, and accessible\n"
        "- Celebrate quality time and making memories outdoors\n"
        "- Invite everyone to join the adventure — no experience needed\n"
        "Good words to include: Relax, Easy, Fun, Ready-to-go, Anticipation, Excitement, Smile, Joy, Family, Play, Hang out, Quality time, Trust, Enjoy."
    )

    prompt = (
        f"{summary_table}\n\n{brand_voice_note}"
        "Given this context, write:\n"
        "1. An optimized YouTube title.\n"
        "2. An optimized long-form YouTube description with 4–7 hashtags at the end.\n"
        "Return with labels 'Title:' and 'Description:'.\n"
        f"Context: {context}\n"
        "\nPlease avoid overused marketing phrases or buzzwords like 'dive in,' 'game changing,' 'next-level,' or 'must-see.' "
        "Use fresh, natural language that reflects the brand’s personality and sounds like something a real person would say."
    )
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.77
    )
    text = res.choices[0].message.content.strip()
    title_match = re.search(r"[Tt]itle[:\-]*\s*(.+)", text)
    desc_match = re.search(r"[Dd]escription[:\-]*\s*(.+)", text, re.DOTALL)
    opt_title = title_match.group(1).strip() if title_match else ""
    opt_desc = desc_match.group(1).strip() if desc_match else ""
    return opt_title, opt_desc

def extract_keywords(description):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform([description])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix[0].toarray().flatten()
    sorted_indices = scores.argsort()[::-1]
    keywords = [feature_names[i] for i in sorted_indices[:8] if scores[i] > 0]
    return keywords

def generate_youtube_tags(description):
    tag_prompt = (
        f"Given the following YouTube description:\n\n{description}\n\n"
        "Suggest 8–12 comma-separated YouTube tags that maximize SEO and are relevant to the content."
    )
    tag_res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": tag_prompt}],
        temperature=0.7
    )
    raw_tags = tag_res.choices[0].message.content
    tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    return tags

def generate_natural_language_analysis(channel_name, video_title, video_desc, opt_title, opt_desc, keywords, tags, user_answers):
    prompt = (
        f"Channel: {channel_name}\n"
        f"Original Video Title: {video_title}\n"
        f"Original Description: {video_desc}\n"
        f"Optimized Title: {opt_title}\n"
        f"Optimized Description: {opt_desc}\n"
        f"Top Keywords: {', '.join(keywords)}\n"
        f"YouTube Tags: {', '.join(tags)}\n"
        f"User Answers: {user_answers}\n"
        "\nWrite a concise, sectioned analysis in this format:\n"
        "Niche: (<2 sentences, what the channel/video is about and industry)\n"
        "Audience: (<2 sentences, target viewer age/interests)\n"
        "Strategy/Reasoning: (<2 sentences, what drove your choices for title/desc/keywords/tags)\n"
        "Key Opportunities: (<1–2 sentences, quick tips for maximizing SEO or growth for this video)\n"
        "SEO Tips: (<1–2 sentences, one or two specific SEO tips for this niche or topic)\n"
        "Each section should be on its own line, no bullet points, easy for a business user to skim."
    )
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.70
    )
    return res.choices[0].message.content.strip()

# def score_frame(frame, frame_idx=None):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     brightness = np.mean(gray)
#     sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
#     contrast = np.std(gray)

#     score = 0.5 * sharpness + 0.3 * brightness + 0.2 * contrast

#     return {
#         "frame": frame,
#         "index": frame_idx,
#         "score": score,
#         "components": {
#             "sharpness": sharpness,
#             "brightness": brightness,
#             "contrast": contrast
#         }
#     }


# def extract_custom_thumbnails(video_path, output_dir="thumbnails", num_frames_to_score=30):
#     os.makedirs(output_dir, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Failed to open video file: {video_path}")

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_indices = np.linspace(0, total_frames - 1, num=num_frames_to_score, dtype=int)
#     print(f"⚙️ Sampling {len(frame_indices)} frames from video (first 5 indices: {frame_indices[:5]})")

#     # ✅ Preload frames into memory (before threading)
#     frames = []
#     for idx in frame_indices:
#         print(f"🔍 Attempting to grab frame at index {idx}/{total_frames}")
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print(f"⚠️ Failed to read frame at index {idx}")
#             continue
#         frames.append((idx, frame))
#     cap.release()

#     # 🧠 Define scoring logic
#     def score_frame(frame, frame_idx=None):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         brightness = np.mean(gray)
#         sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
#         contrast = np.std(gray)

#         score = 0.5 * sharpness + 0.3 * brightness + 0.2 * contrast

#         return {
#             "frame": frame,
#             "index": frame_idx,
#             "score": score,
#             "components": {
#                 "sharpness": sharpness,
#                 "brightness": brightness,
#                 "contrast": contrast
#             }
#         }

#     # ⚡ Multithreaded scoring
#     def process_frame(data):
#         idx, frame = data
#         try:
#             frame = cv2.resize(frame, (640, 360))
#             return score_frame(frame, frame_idx=idx)
#         except Exception as e:
#             print(f"❌ Error processing frame {idx}: {e}")
#             return None

#     scored_frames = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
#         futures = [executor.submit(process_frame, f) for f in frames]
#         for future in concurrent.futures.as_completed(futures):
#             result = future.result()
#             if result:
#                 scored_frames.append(result)

#     if not scored_frames:
#         raise ValueError("❌ No frames were successfully scored.")

#     # 🎯 Best + Diverse Selection
#     best_score = max(scored_frames, key=lambda x: x['score'])
#     best_brightness = max(scored_frames, key=lambda x: x['components']['brightness'])
#     best_contrast = max(scored_frames, key=lambda x: x['components']['contrast'])
#     best_sharpness = max(scored_frames, key=lambda x: x['components']['sharpness'])
#     chosen = {id(f): f for f in [best_score, best_brightness, best_contrast, best_sharpness]}

#     diverse_candidates = [f for f in scored_frames if id(f) not in chosen]
#     diverse_scores = []
#     for f in diverse_candidates:
#         c = f["components"]
#         score = (
#             0.4 * (np.isclose(c["sharpness"], 30, atol=10)) +
#             0.3 * (c["brightness"] < 120) +
#             0.3 * (c["contrast"] < 55)
#         )
#         diverse_scores.append((score, f))

#     diverse_scores.sort(key=lambda x: -x[0])
#     diverse_frames = [item[1] for item in diverse_scores[:4]]
#     all_selected = list(chosen.values()) + diverse_frames

#     # 💾 Save selected thumbnails
#     thumbnails = []
#     for i, item in enumerate(all_selected):
#         c = item["components"]
#         print(f"Frame {item['index']}: Score={item['score']:.2f} | "
#               f"Sharpness={c['sharpness']:.1f}, Brightness={c['brightness']:.1f}, "
#               f"Contrast={c['contrast']:.1f}")
#         filename = f"custom_frame_{i+1}_{uuid.uuid4().hex[:6]}.jpg"
#         filepath = os.path.join(output_dir, filename)
#         cv2.imwrite(filepath, item["frame"])
#         thumbnails.append(filepath)

#     return thumbnails

# def score_frame(frame, frame_idx=None):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     brightness = np.mean(gray)
#     sharpness = np.abs(np.diff(gray.astype(np.int16), axis=0)).mean()
#     contrast = np.std(gray)

#     score = 0.5 * sharpness + 0.3 * brightness + 0.2 * contrast

#     return {
#         "frame": frame,
#         "index": frame_idx,
#         "score": score,
#         "components": {
#             "sharpness": sharpness,
#             "brightness": brightness,
#             "contrast": contrast
#         }
#     }

# def extract_custom_thumbnails(video_path, output_dir="thumbnails", num_frames_to_score=20):
#     os.makedirs(output_dir, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Failed to open video file: {video_path}")

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_indices = np.linspace(0, total_frames - 1, num=num_frames_to_score, dtype=int)
#     frame_indices_set = set(frame_indices)
#     print(f"⚙️ Sampling {len(frame_indices)} frames from video (first 5 indices: {frame_indices[:5]})")

#     frames = []
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_count in frame_indices_set:
#             frames.append((frame_count, frame))
#         frame_count += 1
#     cap.release()

#     def score_frame(frame, frame_idx=None):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         brightness = np.mean(gray)
#         sharpness = np.abs(np.diff(gray.astype(np.int16), axis=0)).mean()  # faster sharpness
#         contrast = np.std(gray)

#         score = 0.5 * sharpness + 0.3 * brightness + 0.2 * contrast
#         return {
#             "frame": frame,
#             "index": frame_idx,
#             "score": score,
#             "components": {
#                 "sharpness": sharpness,
#                 "brightness": brightness,
#                 "contrast": contrast
#             }
#         }

#     scored_frames = []
#     for idx, frame in frames:
#         try:
#             frame = cv2.resize(frame, (320, 180))
#             scored = score_frame(frame, frame_idx=idx)
#             scored_frames.append(scored)
#         except Exception as e:
#             print(f"❌ Error scoring frame {idx}: {e}")

#     if not scored_frames:
#         raise ValueError("❌ No frames were successfully scored.")

#     # Best + Diverse Selection
#     best_score = max(scored_frames, key=lambda x: x['score'])
#     best_brightness = max(scored_frames, key=lambda x: x['components']['brightness'])
#     best_contrast = max(scored_frames, key=lambda x: x['components']['contrast'])
#     best_sharpness = max(scored_frames, key=lambda x: x['components']['sharpness'])
#     chosen = {id(f): f for f in [best_score, best_brightness, best_contrast, best_sharpness]}

#     diverse_candidates = [f for f in scored_frames if id(f) not in chosen]
#     diverse_scores = []
#     for f in diverse_candidates:
#         c = f["components"]
#         score = (
#             0.4 * (np.isclose(c["sharpness"], 30, atol=10)) +
#             0.3 * (c["brightness"] < 120) +
#             0.3 * (c["contrast"] < 55)
#         )
#         diverse_scores.append((score, f))

#     diverse_scores.sort(key=lambda x: -x[0])
#     diverse_frames = [item[1] for item in diverse_scores[:4]]
#     all_selected = list(chosen.values()) + diverse_frames

#     thumbnails = []
#     for i, item in enumerate(all_selected):
#         c = item["components"]
#         print(f"Frame {item['index']}: Score={item['score']:.2f} | "
#               f"Sharpness={c['sharpness']:.1f}, Brightness={c['brightness']:.1f}, "
#               f"Contrast={c['contrast']:.1f}")
#         filename = f"custom_frame_{i+1}_{uuid.uuid4().hex[:6]}.jpg"
#         filepath = os.path.join(output_dir, filename)
#         cv2.imwrite(filepath, item["frame"])
#         thumbnails.append(filepath)

#     return thumbnails

# def extract_custom_thumbnails(video_path, output_dir="thumbnails"):
#     os.makedirs(output_dir, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         raise ValueError(f"Failed to open video file: {video_path}")
    
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     thumbnails = []
#     percentages = [i / 100 for i in range(5, 100, 10)]
    
#     for i, percentage in enumerate(percentages):
#         frame_number = int(percentage * total_frames)
#         # Seek directly to the frame
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
#         ret, frame = cap.read()
#         if ret:
#             resized = cv2.resize(frame, (320, 180))
#             filename = f"thumb_{i+1}_{uuid.uuid4().hex[:6]}.jpg"
#             filepath = os.path.join(output_dir, filename)
#             cv2.imwrite(filepath, resized)
#             thumbnails.append(filepath)
    
#     cap.release()
#     return thumbnails

def extract_thumbnails_ffmpeg(video_path, output_dir="thumbnails", max_width=None):
    """
    Ultra-fast thumbnail extraction using FFmpeg - constant time regardless of video length.
    FFmpeg can seek to exact timestamps without reading through the entire video.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video duration using ffprobe (very fast)
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_format', video_path
        ], capture_output=True, text=True, check=True)
        
        duration = float(json.loads(result.stdout)['format']['duration'])
    except Exception as e:
        print(f"Error getting video duration: {e}")
        # Fallback to OpenCV method
        return extract_custom_thumbnails_opencv_optimized(video_path, output_dir, max_width)
    
    print(f"Video duration: {duration:.2f} seconds")
    
    # Calculate timestamps (5%, 15%, 25%, ..., 95%)
    percentages = [i / 100 for i in range(5, 100, 10)]
    timestamps = [duration * p for p in percentages]
    
    thumbnails = []
    
    for i, timestamp in enumerate(timestamps):
        filename = f"thumb_{i+1}_{uuid.uuid4().hex[:6]}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # FFmpeg command to extract frame at exact timestamp
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite existing files
            '-ss', str(timestamp),  # Seek to timestamp
            '-i', video_path,  # Input video
            '-vframes', '1',  # Extract only 1 frame
            '-q:v', '2',  # High quality (1-31, lower = better)
        ]
        
        # Add width scaling if specified (maintains aspect ratio)
        if max_width:
            cmd.extend(['-vf', f'scale={max_width}:-1'])
        
        cmd.append(filepath)  # Output file
        
        try:
            # Run FFmpeg command
            subprocess.run(cmd, capture_output=True, check=True, timeout=10)
            thumbnails.append(filepath)
            print(f"✅ Extracted thumbnail {i+1}/{len(timestamps)} at {timestamp:.2f}s")
            
        except subprocess.TimeoutExpired:
            print(f"⚠️ Timeout extracting frame at {timestamp:.2f}s")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error extracting frame at {timestamp:.2f}s: {e}")
    
    return thumbnails

def extract_custom_thumbnails_opencv_optimized(video_path, output_dir="thumbnails", max_width=None):
    """
    Optimized OpenCV version as fallback - uses strategic frame positioning.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    # Set properties for fastest seeking
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {total_frames} frames, {fps:.2f} fps")
    
    # Calculate target frames
    percentages = [i / 100 for i in range(5, 100, 10)]
    target_frames = [int(p * total_frames) for p in percentages]
    
    thumbnails = []
    
    for i, frame_num in enumerate(target_frames):
        # Use timestamp-based seeking for better accuracy
        timestamp_ms = (frame_num / fps) * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Could not read frame {frame_num}")
            continue
        
        # Resize if max_width specified
        if max_width:
            height, width = frame.shape[:2]
            if width > max_width:
                scale = max_width / width
                new_height = int(height * scale)
                frame = cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        filename = f"thumb_{i+1}_{uuid.uuid4().hex[:6]}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # High quality JPEG
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumbnails.append(filepath)
        
        print(f"✅ Extracted thumbnail {i+1}/{len(target_frames)}")
    
    cap.release()
    return thumbnails

def check_ffmpeg_available():
    """Check if FFmpeg is available on the system."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def extract_custom_thumbnails(video_path, output_dir="thumbnails", max_width=None):
    """
    Main function that chooses the best extraction method.
    Defaults to FFmpeg for speed, falls back to OpenCV.
    """
    print("🎬 Starting thumbnail extraction...")
    
    if check_ffmpeg_available():
        print("✅ Using FFmpeg for ultra-fast extraction")
        try:
            return extract_thumbnails_ffmpeg(video_path, output_dir, max_width)
        except Exception as e:
            print(f"⚠️ FFmpeg failed, falling back to OpenCV: {e}")
            return extract_custom_thumbnails_opencv_optimized(video_path, output_dir, max_width)
    else:
        print("⚠️ FFmpeg not available, using optimized OpenCV")
        return extract_custom_thumbnails_opencv_optimized(video_path, output_dir, max_width)

def clean_directories(*dirs):
    for d in dirs:
        for filename in os.listdir(d):
            filepath = os.path.join(d, filename)
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"❌ Could not delete {filepath}: {e}")

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/description", methods=["GET", "POST"])
def description():
    try:
        step = request.form.get("step", "entry")
        channel_name = request.form.get("channel_name", "")
        video_title = request.form.get("video_title", "")
        video_desc = request.form.get("video_desc", "")
        improved_message = None

        if request.method == "POST" and step == "entry":
            if "improve_prompt" in request.form:
                improved_desc = improve_description_prompt(channel_name, video_title, video_desc)
                improved_message = "Description improved!"
                video_desc = improved_desc
                return render_template("description.html", step="entry", channel_name=channel_name, video_title=video_title, video_desc=video_desc, improved_message=improved_message)
            
            if "proceed_questions" in request.form:
                questions = generate_questions(channel_name, video_title, video_desc)
                return render_template("description.html", step="questions", channel_name=channel_name, video_title=video_title, video_desc=video_desc, questions=questions)

        elif request.method == "POST" and step == "questions":
            questions = request.form.getlist("question")
            user_answers = [request.form.get(f"answer_{i}", "") for i in range(len(questions))]
            answers_joined = "; ".join(user_answers)

            try:
                keywords_search = f"{video_title} {video_desc}"
                similar_titles = get_similar_titles(keywords_search)
            except Exception:
                similar_titles = []

            optimized_title, optimized_desc = optimize_title_and_desc(channel_name, video_title, video_desc, similar_titles, answers_joined)
            keywords = extract_keywords(optimized_desc)
            tags = generate_youtube_tags(optimized_desc)
            analysis = generate_natural_language_analysis(channel_name, video_title, video_desc, optimized_title, optimized_desc, keywords, tags, answers_joined)

            return render_template("description.html", step="results", channel_name=channel_name, video_title=video_title, video_desc=video_desc, optimized_title=optimized_title, optimized_desc=optimized_desc, keywords=keywords, tags=tags, analysis=analysis)

        return render_template("description.html", step="entry", channel_name=channel_name, video_title=video_title, video_desc=video_desc)

    except Exception as e:
        print(f"Error in /description: {e}")
        return "An error occurred", 500
    
    
# @app.route("/thumbnails", methods=["GET", "POST"])
# def thumbnails():
#     thumbnails = []
#     message = ""

#     if request.method == "POST":
#         # 🧹 Clean previous uploads and thumbnails
#         clean_directories(app.config["UPLOAD_FOLDER"], THUMBNAIL_FOLDER)

#         video = request.files.get("video")
#         if video and video.filename.endswith((".mp4", ".mov", ".avi", ".mkv")):
#             filename = secure_filename(video.filename)
#             upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#             video.save(upload_path)
#             print("✅ Video upload saved to disk.")

#             try:
#                 scored_paths = extract_custom_thumbnails(upload_path, output_dir=THUMBNAIL_FOLDER)
#                 for path in scored_paths:
#                     rel_path = os.path.relpath(path, "static")
#                     thumbnails.append({
#                         "path": rel_path,
#                         "filename": os.path.basename(path)
#                     })
#                 message = "✅ Thumbnails successfully generated!"
#             except Exception as e:
#                 message = f"❌ Error during thumbnail generation: {e}"

#     return render_template("thumbnails.html", thumbnails=thumbnails, message=message)

@app.route("/thumbnails", methods=["GET", "POST"])
def thumbnails():
    thumbnails = []
    message = ""
    
    if request.method == "POST":
        # Clean previous uploads and thumbnails
        clean_directories(app.config["UPLOAD_FOLDER"], THUMBNAIL_FOLDER)
        
        video = request.files.get("video")
        if video and video.filename.endswith((".mp4", ".mov", ".avi", ".mkv")):
            filename = secure_filename(video.filename)
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            video.save(upload_path)
            print("✅ Video upload saved to disk.")
            
            try:
                # Extract thumbnails - should be constant time!
                # Full resolution:
                scored_paths = extract_custom_thumbnails(upload_path, output_dir=THUMBNAIL_FOLDER)
                
                # Or with width limit (maintains aspect ratio):
                # scored_paths = extract_custom_thumbnails(upload_path, output_dir=THUMBNAIL_FOLDER, max_width=1920)
                
                for path in scored_paths:
                    rel_path = os.path.relpath(path, "static")
                    thumbnails.append({
                        "path": rel_path,
                        "filename": os.path.basename(path)
                    })
                
                message = f"✅ {len(scored_paths)} thumbnails generated successfully!"
                
            except Exception as e:
                message = f"❌ Error during thumbnail generation: {e}"
                print(f"Error details: {e}")
        else:
            message = "❌ Please upload a valid video file"
    
    return render_template("thumbnails.html", thumbnails=thumbnails, message=message)



@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(app.config["THUMBNAIL_FOLDER"], filename, as_attachment=True)


@app.route("/test-one-frame")
def test_one_frame():
    import cv2
    import numpy as np

    video_path = "video2.mp4"  # or wherever you're uploading videos
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "❌ Failed to open video"

    ret, frame = cap.read()
    if not ret or frame is None:
        return "❌ Failed to read first frame"

    frame = cv2.resize(frame, (640, 360))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = np.std(gray)

    return (
        f"✅ Success!<br>"
        f"Brightness: {brightness:.2f}<br>"
        f"Sharpness: {sharpness:.2f}<br>"
        f"Contrast: {contrast:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)


