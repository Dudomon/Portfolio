using Abp.Domain.Entities;
using Abp.Domain.Entities.Auditing;
using System.ComponentModel.DataAnnotations.Schema;

namespace WarehouseManagement.Entities
{
    public class OrderLine : FullAuditedEntity<long>, IMustHaveTenant
    {
        public int TenantId { get; set; }

        public long OrderId { get; set; }

        [ForeignKey("OrderId")]
        public Order Order { get; set; } = null!;

        public long InventoryItemId { get; set; }

        [ForeignKey("InventoryItemId")]
        public InventoryItem InventoryItem { get; set; } = null!;

        public int QuantityOrdered { get; set; }
        public int QuantityPicked { get; set; }
        public int QuantityPacked { get; set; }
        public int QuantityShipped { get; set; }

        public decimal UnitPrice { get; set; }
        public decimal TotalPrice => QuantityOrdered * UnitPrice;
    }
}
