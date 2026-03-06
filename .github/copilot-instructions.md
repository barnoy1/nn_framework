---
name: Project Architecture & Quality Rules
description: Strict enforcement of SOLID, file length, folder hierarchy, and modern development best practices.
---

# Global Constraints

## 1. SOLID Principles Enforcement

- **Single Responsibility:** Every class or module must have one, and only one, reason to change. If a function does two things, split it.
- **Open/Closed:** Code should be open for extension but closed for modification. Use interfaces and abstract classes to allow behavior changes without editing core logic.
- **Liskov Substitution:** Subtypes must be substitutable for their base types without breaking the application.
- **Interface Segregation:** Do not force implementation of unused methods. Create small, specific interfaces.
- **Dependency Inversion:** Depend on abstractions, not concretions. Always use Dependency Injection (DI) for services and repositories.

## 2. File Size Limits (Hard Constraint)

- **Max Lines:** No file shall exceed **200 lines** of code.
- **Action:** If a file approaches 180 lines, you MUST suggest a refactor strategy to split the logic into smaller, composable units or helper utility files.
- **Logic Density:** Favor composition over inheritance to keep files lean.

## 3. Directory & Hierarchy Rules

- **Folder Limit:** No single directory should contain more than **5 files**.
- **Nesting:** If a sixth file is required, create a logically named sub-folder (e.g., `/components/buttons/` instead of just `/components/`).
- **Structure:** Maintain a shallow but wide tree. If a folder gets crowded, group related files into a new sub-module.

---

# Best Practices & Paradigms

## 4. Clean Code Standards

- **Naming:** Use intention-revealing names. Variables, functions, and classes must clearly describe their purpose without needing a comment.
  - ❌ `const d = new Date()`
  - ✅ `const createdAt = new Date()`
- **Functions:** Keep functions small — ideally under 20 lines. A function should do one thing and do it well.
- **No Magic Numbers/Strings:** Extract literals into named constants.
  - ❌ `if (status === 3)`
  - ✅ `if (status === OrderStatus.CANCELLED)`
- **Avoid Deep Nesting:** Maximum 2–3 levels of nesting. Use early returns (guard clauses) to flatten logic.
- **Comments:** Code should be self-documenting. Only use comments to explain *why*, never *what*.

## 5. Design Patterns

Apply established patterns where appropriate — never over-engineer, but always recognize when a pattern solves the problem cleanly:

- **Creational**
  - *Factory / Factory Method:* Use when object creation logic is complex or varies by type.
  - *Builder:* Use for constructing complex objects step-by-step (e.g., query builders, form schemas).
  - *Singleton:* Use sparingly — only for truly shared, stateless services (e.g., logger, config). Avoid for anything stateful.

- **Structural**
  - *Adapter:* Use to wrap third-party libraries or external APIs behind your own interface, so they can be swapped without touching core logic.
  - *Facade:* Use to expose a simplified interface over a complex subsystem.
  - *Decorator:* Use to extend behavior without modifying the original class (aligns with Open/Closed).

- **Behavioral**
  - *Strategy:* Use when behavior needs to be swappable at runtime (e.g., different payment processors, sort algorithms).
  - *Observer / Event Emitter:* Use for decoupled, reactive communication between modules.
  - *Repository:* Always abstract data-access logic behind a repository interface. Controllers and services must never query a database directly.
  - *Command:* Use to encapsulate actions as objects, enabling undo, queuing, or logging.

## 6. Functional Programming Principles

Where the language permits, prefer functional patterns for data transformation logic:

- **Pure Functions:** Functions must not produce side effects. Given the same input, always return the same output.
- **Immutability:** Never mutate objects or arrays in place. Use spread operators, `Object.assign`, or immutable data libraries.
  - ❌ `user.name = "Alice"`
  - ✅ `const updatedUser = { ...user, name: "Alice" }`
- **Composition over Inheritance:** Build behavior by composing small, pure functions rather than extending class hierarchies.
- **Avoid Shared State:** Side effects and shared mutable state are the primary source of bugs. Isolate state mutations to clearly defined boundaries (e.g., a state manager or a specific service layer).

## 7. Error Handling

- **Never swallow errors silently.** Every `catch` block must either handle the error, log it with context, or re-throw it.
- **Use typed/custom errors** to distinguish application errors from unexpected system errors.
- **Fail fast:** Validate inputs at the boundary of your system (API layer, service entry points). Do not let invalid data propagate deep into business logic.
- **Result types over exceptions** (where idiomatic): For expected failure states, consider returning a `Result<T, E>` or equivalent rather than throwing.

## 8. Testing Standards

- **Unit Tests:** Every pure function and service method must have unit tests. Mock all external dependencies.
- **Integration Tests:** Cover critical paths end-to-end (e.g., API route → service → repository).
- **Test Naming:** Follow the pattern `should [expected behavior] when [condition]`.
  - ✅ `should return 404 when user is not found`
- **AAA Structure:** Every test must follow Arrange → Act → Assert.
- **No Logic in Tests:** Tests must be simple and declarative. If a test needs a loop or conditional, something is wrong with the design.
- **Coverage:** Aim for ≥80% coverage on business logic. Do not chase 100% — test behavior, not implementation.

## 9. API & Interface Design

- **RESTful conventions:** Use proper HTTP verbs and status codes. Never return `200 OK` for an error.
- **Versioning:** Always version public APIs (e.g., `/api/v1/`).
- **DTOs (Data Transfer Objects):** Never expose raw database models to the API layer. Use dedicated DTO/response objects.
- **Validation at the boundary:** Validate and sanitize all incoming data at the controller/handler level before it reaches business logic.
- **Pagination:** All list endpoints must support cursor-based or offset pagination. Never return unbounded arrays.

## 10. Performance Awareness

- **Avoid premature optimization**, but always be aware of algorithmic complexity (Big-O).
- **N+1 Query Prevention:** When fetching related data, always use batch loading, eager loading, or equivalent patterns.
- **Caching:** Cache at the appropriate layer (HTTP, service, or data layer). Document cache invalidation strategy wherever caching is applied.
- **Async/Non-blocking:** I/O operations (DB queries, HTTP calls, file reads) must always be asynchronous. Never block the event loop or main thread.

## 11. Security Defaults

- **Never trust user input.** Sanitize and validate everything coming from outside the system.
- **No secrets in code.** All credentials, API keys, and environment-specific values must live in environment variables, never in source files.
- **Principle of Least Privilege:** Services, DB users, and API tokens should only have the minimum permissions they need.
- **Dependency hygiene:** Regularly audit and update dependencies. Do not pull in large libraries for trivial utilities.

---

# Interaction Rules

- Before providing code, verify if the addition will push the file over the **200-line limit**.
- If the request violates SOLID (e.g., adding "God Object" functionality), **warn the user** and propose a decoupled alternative.
- Always organize new exports into a clear folder structure that respects the **5-file limit**.
- When a design pattern would improve a proposed solution, **name it explicitly** and explain why it applies.
- When suggesting a refactor, always provide the **before/after structure**, not just the concept.
- Flag any code that introduces shared mutable state, magic values, or silent error handling.