using Abp.Domain.Entities;
using Abp.Domain.Entities.Auditing;
using System.ComponentModel.DataAnnotations;

namespace WarehouseManagement.Entities
{
    public class Warehouse : FullAuditedEntity<long>, IMustHaveTenant
    {
        public int TenantId { get; set; }

        [Required]
        [StringLength(WarehouseManagementConsts.MaxCodeLength)]
        public string Code { get; set; } = string.Empty;

        [Required]
        [StringLength(WarehouseManagementConsts.MaxNameLength)]
        public string Name { get; set; } = string.Empty;

        [StringLength(WarehouseManagementConsts.MaxDescriptionLength)]
        public string? Description { get; set; }

        [Required]
        [StringLength(500)]
        public string Address { get; set; } = string.Empty;

        [Required]
        [StringLength(100)]
        public string City { get; set; } = string.Empty;

        [Required]
        [StringLength(100)]
        public string State { get; set; } = string.Empty;

        [Required]
        [StringLength(20)]
        public string ZipCode { get; set; } = string.Empty;

        [Required]
        [StringLength(100)]
        public string Country { get; set; } = string.Empty;

        public decimal TotalArea { get; set; }
        public decimal StorageCapacity { get; set; }
        public bool IsActive { get; set; }

        public ICollection<WarehouseZone> Zones { get; set; } = new List<WarehouseZone>();
        public ICollection<InventoryItem> InventoryItems { get; set; } = new List<InventoryItem>();
    }
}
