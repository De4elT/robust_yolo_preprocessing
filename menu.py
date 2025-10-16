import os
import sys


def list_images_in_folder(folder_path):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    try:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        if not files:
            print("No image files found in:", folder_path)
            return []
        print("Available images in", folder_path)
        for idx, file in enumerate(files):
            print(f"{idx + 1}. {file}")
        return files
    except FileNotFoundError:
        print("Folder not found:", folder_path)
        return []

def main_menu():
    while True:
        print("\n====================")
        print("   MAIN MENU")
        print("====================")
        print("1. Test LPB + YOLOv7 on a single image")
        print("2. Test LPB + YOLOv7 on a folder of images")
        print("3. Compare LPB vs classic blur")
        print("4. Select YOLOv7 model variant")
        print("5. Exit")
        print("====================")

        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            image_folder = os.path.join("data", "images")
            list_images_in_folder(image_folder)
        elif choice == "2":
            print("Batch testing not implemented yet.")
        elif choice == "3":
            print("Comparison not implemented yet.")
        elif choice == "4":
            print("Model selection not implemented yet.")
        elif choice == "5":
            print("Returning to menu...")
        else:
            print("Invalid choice. Please try again.")

        input("\nPress Enter to return to the menu...")

if __name__ == "__main__":
    main_menu()