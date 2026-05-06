import os

folder = "/home/ec2-user/experiment"  # replace with your actual folder

# List all .mp3 files (sorted to keep consistent order)
mp3_files = sorted([f for f in os.listdir(folder) if f.endswith(".mp3")])

# Rename to 1.mp3, 2.mp3, ...
for idx, filename in enumerate(mp3_files, start=1):
    new_name = f"{idx}.mp3"
    src = os.path.join(folder, filename)
    dst = os.path.join(folder, new_name)
    os.rename(src, dst)
    print(f"Renamed '{filename}' to '{new_name}'")
