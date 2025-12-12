using Abp.Domain.Entities;
using Abp.Domain.Entities.Auditing;
using System.ComponentModel.DataAnnotations;

namespace WarehouseManagement.Entities
{
    public class Order : FullAuditedEntity<long>, IMustHaveTenant
    {
        public int TenantId { get; set; }

        [Required]
        [StringLength(WarehouseManagementConsts.MaxCodeLength)]
        public string OrderNumber { get; set; } = string.Empty;

        public long WarehouseId { get; set; }
        public Warehouse Warehouse { get; set; } = null!;

        public OrderType Type { get; set; }
        public OrderStatus Status { get; set; }

        [Required]
        [StringLength(WarehouseManagementConsts.MaxNameLength)]
        public string CustomerName { get; set; } = string.Empty;

        [StringLength(500)]
        public string? ShippingAddress { get; set; }

        public DateTime OrderDate { get; set; }
        public DateTime? ExpectedDeliveryDate { get; set; }
        public DateTime? ActualDeliveryDate { get; set; }

        public decimal TotalValue { get; set; }

        public ICollection<OrderLine> OrderLines { get; set; } = new List<OrderLine>();
        public ICollection<PickingTask> PickingTasks { get; set; } = new List<PickingTask>();
    }

    public enum OrderType
    {
        Inbound = 1,
        Outbound = 2,
        Transfer = 3
    }

    public enum OrderStatus
    {
        Pending = 1,
        Confirmed = 2,
        Picking = 3,
        Packing = 4,
        Shipped = 5,
        Delivered = 6,
        Cancelled = 7
    }
}
