using FleetManagement.Application.DTOs;
using FluentValidation;

namespace FleetManagement.Application.Validators;

public class CreateVehicleValidator : AbstractValidator<CreateVehicleDto>
{
    public CreateVehicleValidator()
    {
        RuleFor(x => x.PlateNumber)
            .NotEmpty().WithMessage("Plate number is required")
            .MaximumLength(20).WithMessage("Plate number cannot exceed 20 characters");

        RuleFor(x => x.Brand)
            .NotEmpty().WithMessage("Brand is required")
            .MaximumLength(100).WithMessage("Brand cannot exceed 100 characters");

        RuleFor(x => x.Model)
            .NotEmpty().WithMessage("Model is required")
            .MaximumLength(100).WithMessage("Model cannot exceed 100 characters");

        RuleFor(x => x.Year)
            .InclusiveBetween(1900, DateTime.UtcNow.Year + 1)
            .WithMessage("Year must be between 1900 and next year");

        RuleFor(x => x.VinNumber)
            .NotEmpty().WithMessage("VIN number is required")
            .MaximumLength(50).WithMessage("VIN number cannot exceed 50 characters");

        RuleFor(x => x.LoadCapacity)
            .GreaterThan(0).WithMessage("Load capacity must be greater than 0");

        RuleFor(x => x.CurrentMileage)
            .GreaterThanOrEqualTo(0).WithMessage("Current mileage cannot be negative");

        RuleFor(x => x.FuelConsumption)
            .GreaterThan(0).WithMessage("Fuel consumption must be greater than 0");
    }
}
