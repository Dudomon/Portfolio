using Abp.Domain.Entities;
using Abp.Domain.Entities.Auditing;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace WarehouseManagement.Entities
{
    public class InventoryLocation : FullAuditedEntity<long>, IMustHaveTenant
    {
        public int TenantId { get; set; }

        [Required]
        [StringLength(WarehouseManagementConsts.MaxCodeLength)]
        public string Code { get; set; } = string.Empty;

        public long ZoneId { get; set; }

        [ForeignKey("ZoneId")]
        public WarehouseZone Zone { get; set; } = null!;

        [StringLength(20)]
        public string? Aisle { get; set; }

        [StringLength(20)]
        public string? Rack { get; set; }

        [StringLength(20)]
        public string? Shelf { get; set; }

        [StringLength(20)]
        public string? Bin { get; set; }

        public bool IsAvailable { get; set; }
        public int Capacity { get; set; }
        public int CurrentOccupancy { get; set; }
    }
}
