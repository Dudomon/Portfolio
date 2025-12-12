using Abp.Domain.Entities;
using Abp.Domain.Entities.Auditing;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace WarehouseManagement.Entities
{
    public class InventoryItem : FullAuditedEntity<long>, IMustHaveTenant
    {
        public int TenantId { get; set; }

        [Required]
        [StringLength(WarehouseManagementConsts.MaxCodeLength)]
        public string SKU { get; set; } = string.Empty;

        [Required]
        [StringLength(WarehouseManagementConsts.MaxNameLength)]
        public string ProductName { get; set; } = string.Empty;

        [StringLength(WarehouseManagementConsts.MaxDescriptionLength)]
        public string? Description { get; set; }

        public long WarehouseId { get; set; }

        [ForeignKey("WarehouseId")]
        public Warehouse Warehouse { get; set; } = null!;

        public long? LocationId { get; set; }

        [ForeignKey("LocationId")]
        public InventoryLocation? Location { get; set; }

        public int QuantityOnHand { get; set; }
        public int QuantityReserved { get; set; }
        public int QuantityAvailable => QuantityOnHand - QuantityReserved;
        public int ReorderPoint { get; set; }
        public int ReorderQuantity { get; set; }

        public decimal UnitCost { get; set; }
        public decimal TotalValue => QuantityOnHand * UnitCost;

        [StringLength(50)]
        public string? BatchNumber { get; set; }

        public DateTime? ExpirationDate { get; set; }

        public bool IsActive { get; set; }
    }
}
