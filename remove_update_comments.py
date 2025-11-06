import os

folder_path = r"C:\Users\Ayush Tiwari\OneDrive\Desktop\CarSegmentation\Projects_2"


for root, _, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

print("✅ All update/sync comment lines removed successfully!")
