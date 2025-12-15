# Laravel-Style Architecture

This project has been refactored to follow Laravel-style architectural patterns while maintaining Flask as the web framework.

## Architecture Overview

### 1. Service Providers (`app/providers/`)
- **AppServiceProvider**: Manages dependency injection and service registration
- Centralized service container following Laravel's service provider pattern
- Handles model loading and service initialization

### 2. Services (`app/services/`)
- **PredictionService**: Business logic for text classification predictions
- **ModelService**: Business logic for model management and health checks
- Separates business logic from controllers (Laravel-style service layer)

### 3. Repositories (`app/repositories/`)
- **ModelRepository**: Data access layer for model persistence
- Implements repository pattern for clean separation of data access
- Handles model loading, storage, and retrieval

### 4. Middleware (`app/middleware/`)
- **ErrorHandler**: Centralized error handling middleware
- Decorator-based middleware following Laravel patterns
- Provides consistent error responses

### 5. Configuration (`config/`)
- **app.py**: Centralized configuration management
- Environment-based configuration (development/production)
- Follows Laravel's config structure

## Benefits

1. **Separation of Concerns**: Business logic separated from controllers
2. **Dependency Injection**: Services injected through service provider
3. **Testability**: Easy to mock services and repositories
4. **Maintainability**: Clear structure and responsibilities
5. **Scalability**: Easy to add new services and features

## Project Structure

```
SMS-SPAM-PIT-elect3/
├── app/
│   ├── __init__.py
│   ├── providers/
│   │   ├── __init__.py
│   │   └── app_service_provider.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── prediction_service.py
│   │   └── model_service.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── model_repository.py
│   └── middleware/
│       ├── __init__.py
│       └── error_handler.py
├── config/
│   └── app.py
├── app.py (refactored to use new architecture)
└── ...
```

## Usage

The API endpoints remain the same, but now use the Laravel-style architecture:

- Controllers (routes) delegate to services
- Services contain business logic
- Repositories handle data access
- Service provider manages dependencies

Example flow:
1. Request → Route Handler
2. Route Handler → Service Provider → Service
3. Service → Repository
4. Repository → Returns data
5. Service → Processes data
6. Route Handler → Returns response
