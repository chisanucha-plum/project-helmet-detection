---
name: backend
description: "Use this skill whenever working with Python backend code, FastAPI, or when the user mentions: backend review, Python code style, refactor API, improve code structure, dataclass, type hints, logging, configuration, or any Python/backend-related task. Trigger even when users say 'review this endpoint', 'clean up this code', 'check Python style', or 'improve this module'."
---

# Backend Python Coding Style

Python coding style guidelines focused on simplicity, clarity, and maintainability for FastAPI backend projects.

## Critical rules

- Write code that is easy to read, not overly complex
- Use type hints everywhere
- Use dataclass for data structures
- Never hardcode values - use YAML configuration
- Use logging instead of print statements
- Keep functions under 50 lines
- Always cleanup resources with try-finally
- Use Pathlib instead of string paths

## Workflow

### 1. Core Principles

**1.1. Simplicity First**
- Write code that is easy to read and not overly complex
- Avoid unnecessary abstraction
- Use clear and meaningful variable and function names
- Do not add features that are not being used

**1.2. Clear Module Structure**
Organize code by responsibility: schemas/, utils/, models/, processors/, main.py

**1.3. Use Dataclass for Data Structures**

Use `@dataclass` from dataclasses module with proper type hints and default_factory for mutable defaults

**1.4. Use Type Hints Everywhere**

Add type hints to all function parameters and return values

**1.5. Separate Configuration**
- Use YAML for configuration
- Do not hardcode values in code
- Create a Configuration class to manage config

**1.6. Error Handling and Logging**

Use try-except-finally with proper logging, always cleanup in finally block

**1.7. Reduce Unnecessary Print Statements**

Use logger.info/error instead of print statements, keep output concise

**1.8. Use Meaningful Names**

Use descriptive function and variable names, even if slightly longer

**1.9. Separate Concerns**

Keep responsibilities separated: file operations, formatting, model operations, business logic in separate functions/classes

**1.10. Resource Management**

Use try-finally to always cleanup temp files and resources

**1.11. Concise Docstrings**

Write brief docstrings with Args and Returns sections

**1.12. Private Methods**

Use underscore prefix (_method_name) for internal helper methods

### 2. Things to Avoid

- Abstract factories and complex patterns (unless necessary)
- Unnecessary nested classes
- Magic numbers (use config instead)
- Global variables (pass parameters instead)
- Print statements in production code (use logging)
- Hardcoded paths (use config)
- Functions longer than 50 lines (break into smaller functions)

### 3. Best Practices

- Dataclasses for data structures
- Type hints wherever possible
- YAML configuration for settings
- Logging module instead of print
- Try-finally for resource cleanup
- List comprehension when appropriate
- Pathlib instead of string paths
- Descriptive naming, even if slightly longer

### 4. Code Example Structure

Good code structure separates concerns into modules:
- `schemas/` - Data structures with dataclasses
- `utils/` - Utility functions with type hints
- `processors/` - Business logic classes
- Use logging, try-except-finally, descriptive names
- Keep methods focused and under 50 lines

## Bundled resources

None - This skill contains all necessary guidelines inline.