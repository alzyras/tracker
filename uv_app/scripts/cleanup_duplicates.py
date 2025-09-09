#!/usr/bin/env python3
# uv_app/scripts/cleanup_duplicates.py

"""
Duplicate Cleanup Script

This script helps find and merge duplicate people in the tracking database.
It identifies people with similar faces and allows you to merge them into a single person.
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.management import DuplicateCleaner, PeopleManager


def find_duplicates(cleaner: DuplicateCleaner, threshold: float = 0.3) -> None:
    """Find and display potential duplicates."""
    print(f"Searching for duplicates with similarity threshold: {threshold}")
    print("=" * 60)
    
    duplicates = cleaner.find_duplicates(threshold)
    
    if not duplicates:
        print("No potential duplicates found.")
        return
    
    print(f"Found {len(duplicates)} potential duplicate groups:")
    print()
    
    for i, group in enumerate(duplicates, 1):
        print(f"Group {i}: IDs {group}")
        
        # Show details for each person in the group
        manager = PeopleManager()
        for track_id in group:
            details = manager.get_person_details(track_id)
            if details:
                name = details['name'] or 'Unnamed'
                faces = details['num_face_images']
                print(f"  - ID {track_id}: {name} ({faces} faces)")
        print()


def interactive_merge(cleaner: DuplicateCleaner, duplicates: list) -> None:
    """Interactively merge duplicate groups."""
    manager = PeopleManager()
    
    for i, group in enumerate(duplicates, 1):
        print(f"\nDuplicate Group {i}: IDs {group}")
        print("-" * 40)
        
        # Show details for each person
        for track_id in group:
            details = manager.get_person_details(track_id)
            if details:
                name = details['name'] or 'Unnamed'
                faces = details['num_face_images']
                print(f"  ID {track_id}: {name} ({faces} faces)")
        
        # Ask which person to keep
        while True:
            try:
                keep_id = input(f"\nWhich person ID should be kept? (merge others into this one): ")
                keep_id = int(keep_id)
                if keep_id in group:
                    break
                else:
                    print(f"Please enter one of the IDs from the group: {group}")
            except ValueError:
                print("Please enter a valid ID number")
            except KeyboardInterrupt:
                print("\nSkipping this group...")
                break
        
        if keep_id in group:
            # Confirm merge
            response = input(f"Merge all other people into ID {keep_id}? (y/N): ")
            if response.lower() == 'y':
                if cleaner.merge_duplicates(group, keep_id):
                    print(f"Successfully merged group {i} into person {keep_id}")
                else:
                    print(f"Failed to merge group {i}")
            else:
                print(f"Skipped merging group {i}")


def auto_merge(cleaner: DuplicateCleaner, duplicates: list, keep_strategy: str = "most_faces") -> None:
    """Automatically merge duplicates using a strategy."""
    manager = PeopleManager()
    
    for i, group in enumerate(duplicates, 1):
        print(f"\nProcessing Group {i}: IDs {group}")
        
        # Determine which person to keep based on strategy
        if keep_strategy == "most_faces":
            # Keep person with most face images
            best_id = None
            max_faces = -1
            
            for track_id in group:
                details = manager.get_person_details(track_id)
                if details and details['num_face_images'] > max_faces:
                    max_faces = details['num_face_images']
                    best_id = track_id
        elif keep_strategy == "lowest_id":
            # Keep person with lowest ID
            best_id = min(group)
        else:
            print(f"Unknown strategy: {keep_strategy}")
            continue
        
        if best_id:
            print(f"Keeping ID {best_id} (strategy: {keep_strategy})")
            if cleaner.merge_duplicates(group, best_id):
                print(f"Successfully merged group {i} into person {best_id}")
            else:
                print(f"Failed to merge group {i}")
        else:
            print(f"Could not determine which person to keep in group {i}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clean up duplicate people")
    parser.add_argument("--threshold", "-t", type=float, default=0.3,
                       help="Similarity threshold for detecting duplicates (0.0-1.0)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive mode - ask for each merge")
    parser.add_argument("--auto", "-a", action="store_true",
                       help="Automatic mode - merge without asking")
    parser.add_argument("--strategy", "-s", choices=["most_faces", "lowest_id"], 
                       default="most_faces",
                       help="Strategy for choosing which person to keep in auto mode")
    parser.add_argument("--find-only", "-f", action="store_true",
                       help="Only find duplicates, don't merge")
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: Threshold must be between 0.0 and 1.0")
        return 1
    
    # Initialize cleaner
    cleaner = DuplicateCleaner()
    
    # Find duplicates
    duplicates = cleaner.find_duplicates(args.threshold)
    
    if not duplicates:
        print("No duplicates found.")
        return 0
    
    # Show duplicates
    find_duplicates(cleaner, args.threshold)
    
    if args.find_only:
        return 0
    
    # Merge duplicates
    if args.interactive:
        interactive_merge(cleaner, duplicates)
    elif args.auto:
        auto_merge(cleaner, duplicates, args.strategy)
    else:
        print("\nUse --interactive, --auto, or --find-only to proceed.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
