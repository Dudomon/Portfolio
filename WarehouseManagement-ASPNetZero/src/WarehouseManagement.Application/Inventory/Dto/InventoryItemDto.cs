using Abp.Application.Services.Dto;
using Abp.AutoMapper;
using WarehouseManagement.Entities;

namespace WarehouseManagement.Inventory.Dto
{
    [AutoMapFrom(typeof(InventoryItem))]
    public class InventoryItemDto : FullAuditedEntityDto<long>
    {
        public string SKU { get; set; } = string.Empty;
        public string ProductName { get; set; } = string.Empty;
        public string? Description { get; set; }
        public long WarehouseId { get; set; }
        public string WarehouseName { get; set; } = string.Empty;
        public long? LocationId { get; set; }
        public string? LocationCode { get; set; }
        public int QuantityOnHand { get; set; }
        public int QuantityReserved { get; set; }
        public int QuantityAvailable { get; set; }
        public int ReorderPoint { get; set; }
        public int ReorderQuantity { get; set; }
        public decimal UnitCost { get; set; }
        public decimal TotalValue { get; set; }
        public string? BatchNumber { get; set; }
        public DateTime? ExpirationDate { get; set; }
        public bool IsActive { get; set; }
        public bool NeedsReorder => QuantityAvailable <= ReorderPoint;
    }

    [AutoMapTo(typeof(InventoryItem))]
    public class CreateInventoryItemDto
    {
        public string SKU { get; set; } = string.Empty;
        public string ProductName { get; set; } = string.Empty;
        public string? Description { get; set; }
        public long WarehouseId { get; set; }
        public long? LocationId { get; set; }
        public int QuantityOnHand { get; set; }
        public int ReorderPoint { get; set; }
        public int ReorderQuantity { get; set; }
        public decimal UnitCost { get; set; }
        public string? BatchNumber { get; set; }
        public DateTime? ExpirationDate { get; set; }
    }

    public class AdjustInventoryQuantityInput
    {
        public long InventoryItemId { get; set; }
        public int QuantityAdjustment { get; set; }
        public string? Reason { get; set; }
    }
}
