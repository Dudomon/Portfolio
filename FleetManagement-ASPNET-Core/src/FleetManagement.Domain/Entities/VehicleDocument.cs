using FleetManagement.Domain.Enums;

namespace FleetManagement.Domain.Entities;

public class VehicleDocument : BaseEntity
{
    public Guid VehicleId { get; set; }
    public Vehicle Vehicle { get; set; } = null!;
    public DocumentType Type { get; set; }
    public string DocumentNumber { get; set; } = string.Empty;
    public DateTime IssueDate { get; set; }
    public DateTime ExpiryDate { get; set; }
    public string? IssuingAuthority { get; set; }
    public string? FilePath { get; set; }
    public string? Notes { get; set; }
}
