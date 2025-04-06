# Linked List Implementation in C

This experiment implements a basic linked list data structure in C, using Gradle as the build system.

## Project Structure

```
src/linked-list/
├── README.md           # This file
├── build.gradle        # Gradle build configuration
├── linked_list.h       # Header file with declarations
├── linked_list.c       # Implementation file
└── main.c             # Example usage
```

## Prerequisites

- Gradle (version 8.13 or later)
- C compiler (GCC or Clang)
- Git

## Building and Running

1. Clone the repository
2. Navigate to the experiment directory:
   ```bash
   cd src/linked-list
   ```
3. Build the project:
   ```bash
   ./gradlew build
   ```
4. Run the program:
   ```bash
   ./gradlew run
   ```

## Adding New Features

1. Add function declarations to `linked_list.h`
2. Implement the functions in `linked_list.c`
3. Add test cases to `main.c`
4. Build and test your changes:
   ```bash
   ./gradlew clean build run
   ```

## Code Style Guidelines

- Use consistent indentation (4 spaces)
- Add comments for complex logic
- Follow the existing naming conventions:
  - Function names: camelCase
  - Type names: PascalCase
  - Variable names: snake_case

## Testing

The `main.c` file contains example usage of the linked list. When adding new features:
1. Add test cases to `main.c`
2. Ensure all memory is properly allocated and freed
3. Test edge cases (empty list, single element, etc.)

## Contributing

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Test thoroughly
4. Create a pull request

## License

This project is open source and available under the MIT License. 