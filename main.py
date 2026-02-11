from auth.verify import verify_admin

def main():
    print("Say wake phrase...")
    
    if verify_admin():
        print("Agent Activated")
    else:
        print("Unauthorized attempt.")

if __name__ == "__main__":
    main()