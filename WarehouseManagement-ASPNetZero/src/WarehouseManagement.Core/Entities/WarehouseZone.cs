using Abp.Domain.Entities;
using Abp.Domain.Entities.Auditing;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace WarehouseManagement.Entities
{
    public class WarehouseZone : FullAuditedEntity<long>, IMustHaveTenant
    {
        public int TenantId { get; set; }

        [Required]
        [StringLength(WarehouseManagementConsts.MaxCodeLength)]
        public string Code { get; set; } = string.Empty;

        [Required]
        [StringLength(WarehouseManagementConsts.MaxNameLength)]
        public string Name { get; set; } = string.Empty;

        public long WarehouseId { get; set; }

        [ForeignKey("WarehouseId")]
        public Warehouse Warehouse { get; set; } = null!;

        public ZoneType Type { get; set; }
        public decimal Area { get; set; }
        public int Capacity { get; set; }
        public bool IsActive { get; set; }

        public ICollection<InventoryLocation> Locations { get; set; } = new List<InventoryLocation>();
    }

    public enum ZoneType
    {
        Receiving = 1,
        Storage = 2,
        Picking = 3,
        Packing = 4,
        Shipping = 5,
        Quarantine = 6
    }
}
