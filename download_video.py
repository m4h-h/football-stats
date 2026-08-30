"""
Download a match video from YouTube for analysis.

Usage:
    python download_video.py "https://www.youtube.com/watch?v=Fm-xLR1X4Js&t=16s"
"""
import sys
import os

def main():
    if len(sys.argv) < 2:
        print('Usage: python download_video.py "<youtube_url>"')
        sys.exit(1)

    url = sys.argv[1]
    out_dir = "input_videos"
    os.makedirs(out_dir, exist_ok=True)

    try:
        import yt_dlp
    except ImportError:
        print("yt-dlp not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    ydl_opts = {
        # mp4, best quality up to 1080p (higher res = slower detection, rarely worth it)
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)
        # merge_output_format may change extension to mp4
        base, _ = os.path.splitext(filepath)
        mp4_path = base + ".mp4"
        final_path = mp4_path if os.path.exists(mp4_path) else filepath
        print(f"\nDownloaded to: {final_path}")
        print("Now run: python main.py \"" + final_path + "\"")


if __name__ == "__main__":
    main()