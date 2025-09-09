#!/usr/bin/env python3
# uv_app/scripts/manage_people.py

"""
People Management Script

This script provides a command-line interface for managing tracked people.
It allows you to list, view, rename, and delete people from the tracking database.
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.management import PeopleManager


def list_people(manager: PeopleManager, detailed: bool = False) -> None:
    """List all tracked people."""
    people = manager.list_people()
    
    if not people:
        print("No tracked people found.")
        return
    
    print(f"\nFound {len(people)} tracked people:")
    print("-" * 60)
    
    for person in people:
        print(f"ID: {person['track_id']}")
        print(f"Name: {person['name'] or 'Unnamed'}")
        print(f"Face Images: {person['num_faces']}")
        print(f"Has Encodings: {'Yes' if person['has_encodings'] else 'No'}")
        
        if detailed:
            details = manager.get_person_details(person['track_id'])
            if details:
                print(f"Emotion History: {len(details['emotion_history'])} entries")
                print(f"Phone History: {len(details['phone_history'])} entries")
                print(f"Body Actions: {list(details['body_actions'].keys())}")
        
        print("-" * 60)


def view_person(manager: PeopleManager, track_id: int) -> None:
    """View details of a specific person."""
    details = manager.get_person_details(track_id)
    
    if not details:
        print(f"Person with ID {track_id} not found.")
        return
    
    print(f"\nPerson Details - ID: {track_id}")
    print("=" * 50)
    print(f"Name: {details['name'] or 'Unnamed'}")
    print(f"Face Encodings: {details['num_face_encodings']}")
    print(f"Face Images: {details['num_face_images']}")
    print(f"Data Path: {details['data_path']}")
    
    if details['emotion_history']:
        print(f"\nEmotion History: {details['emotion_history']}")
    
    if details['phone_history']:
        print(f"Phone History: {details['phone_history']}")
    
    if details['body_actions']:
        print(f"Body Actions: {details['body_actions']}")
    
    if details['face_images']:
        print(f"\nFace Images:")
        for i, img in enumerate(details['face_images'][:5], 1):  # Show first 5
            print(f"  {i}. {img}")
        if len(details['face_images']) > 5:
            print(f"  ... and {len(details['face_images']) - 5} more")


def rename_person(manager: PeopleManager, track_id: int, new_name: str) -> None:
    """Rename a person."""
    if manager.rename_person(track_id, new_name):
        print(f"Person {track_id} renamed to '{new_name}' successfully.")
    else:
        print(f"Failed to rename person {track_id}.")


def delete_person(manager: PeopleManager, track_id: int, confirm: bool = False) -> None:
    """Delete a person."""
    if not confirm:
        response = input(f"Are you sure you want to delete person {track_id}? (y/N): ")
        if response.lower() != 'y':
            print("Deletion cancelled.")
            return
    
    if manager.delete_person(track_id):
        print(f"Person {track_id} deleted successfully.")
    else:
        print(f"Failed to delete person {track_id}.")


def view_face_images(manager: PeopleManager, track_id: int) -> None:
    """View face images for a person."""
    images = manager.view_face_images(track_id)
    
    if not images:
        print(f"No face images found for person {track_id}.")
        return
    
    print(f"\nFace images for person {track_id}:")
    for i, img in enumerate(images, 1):
        print(f"{i:2d}. {img}")
    
    # Ask if user wants to view an image
    try:
        choice = input(f"\nEnter image number to view (1-{len(images)}) or press Enter to skip: ")
        if choice.strip():
            img_num = int(choice) - 1
            if 0 <= img_num < len(images):
                manager.display_face_image(track_id, images[img_num])
            else:
                print("Invalid image number.")
    except (ValueError, KeyboardInterrupt):
        print("Skipping image display.")


def cleanup_empty_people(manager: PeopleManager) -> None:
    """Clean up people with no face data."""
    cleaned = manager.cleanup_empty_people()
    print(f"Cleaned up {cleaned} empty people directories.")


def export_person(manager: PeopleManager, track_id: int, output_file: str) -> None:
    """Export person data to JSON file."""
    if manager.export_person_data(track_id, output_file):
        print(f"Person {track_id} data exported to {output_file}")
    else:
        print(f"Failed to export person {track_id} data.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Manage tracked people")
    parser.add_argument("--list", "-l", action="store_true", help="List all people")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed information")
    parser.add_argument("--view", "-v", type=int, help="View specific person by ID")
    parser.add_argument("--rename", "-r", nargs=2, metavar=("ID", "NAME"), 
                       help="Rename person (ID and new name)")
    parser.add_argument("--delete", "-del", type=int, help="Delete person by ID")
    parser.add_argument("--images", "-i", type=int, help="View face images for person ID")
    parser.add_argument("--cleanup", "-c", action="store_true", help="Clean up empty people")
    parser.add_argument("--export", "-e", nargs=2, metavar=("ID", "FILE"), 
                       help="Export person data to JSON file")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = PeopleManager()
    
    # Execute commands
    if args.list:
        list_people(manager, args.detailed)
    elif args.view:
        view_person(manager, args.view)
    elif args.rename:
        track_id, new_name = args.rename
        rename_person(manager, track_id, new_name)
    elif args.delete:
        delete_person(manager, args.delete)
    elif args.images:
        view_face_images(manager, args.images)
    elif args.cleanup:
        cleanup_empty_people(manager)
    elif args.export:
        track_id, output_file = args.export
        export_person(manager, track_id, output_file)
    else:
        # Default: list people
        list_people(manager)


if __name__ == "__main__":
    main()
