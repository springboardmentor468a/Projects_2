import os

folder_path = r"C:\Users\Ayush Tiwari\OneDrive\Desktop\CarSegmentation\Projects_2"

found = False
for root, _, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if "# update for consistency" in line or "# minor sync update" in line:
                        print(f"⚠️ Found in {file}: {line.strip()}")
                        found = True

if not found:
    print("✅ No 'update' or 'sync' comments found! All clean 🎉")
