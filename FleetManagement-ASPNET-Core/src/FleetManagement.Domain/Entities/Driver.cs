using FleetManagement.Domain.Enums;

namespace FleetManagement.Domain.Entities;

public class Driver : BaseEntity
{
    public string FirstName { get; set; } = string.Empty;
    public string LastName { get; set; } = string.Empty;
    public string FullName => $"{FirstName} {LastName}";
    public string Email { get; set; } = string.Empty;
    public string PhoneNumber { get; set; } = string.Empty;
    public string LicenseNumber { get; set; } = string.Empty;
    public DateTime LicenseExpiryDate { get; set; }
    public DriverLicenseCategory LicenseCategory { get; set; }
    public DriverStatus Status { get; set; }
    public DateTime HireDate { get; set; }
    public string Address { get; set; } = string.Empty;
    public string City { get; set; } = string.Empty;
    public string State { get; set; } = string.Empty;
    public string ZipCode { get; set; } = string.Empty;
    public DateTime DateOfBirth { get; set; }
    public string? EmergencyContactName { get; set; }
    public string? EmergencyContactPhone { get; set; }
    public string? Notes { get; set; }

    public ICollection<Route> Routes { get; set; } = new List<Route>();
    public ICollection<DriverDocument> Documents { get; set; } = new List<DriverDocument>();
}
