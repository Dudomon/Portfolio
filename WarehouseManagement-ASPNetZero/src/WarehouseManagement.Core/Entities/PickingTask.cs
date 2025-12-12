using Abp.Domain.Entities;
using Abp.Domain.Entities.Auditing;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace WarehouseManagement.Entities
{
    public class PickingTask : FullAuditedEntity<long>, IMustHaveTenant
    {
        public int TenantId { get; set; }

        [Required]
        [StringLength(WarehouseManagementConsts.MaxCodeLength)]
        public string TaskNumber { get; set; } = string.Empty;

        public long OrderId { get; set; }

        [ForeignKey("OrderId")]
        public Order Order { get; set; } = null!;

        public long? AssignedUserId { get; set; }

        public PickingTaskStatus Status { get; set; }

        public DateTime? StartedAt { get; set; }
        public DateTime? CompletedAt { get; set; }

        public int Priority { get; set; }

        [StringLength(WarehouseManagementConsts.MaxDescriptionLength)]
        public string? Notes { get; set; }
    }

    public enum PickingTaskStatus
    {
        Pending = 1,
        Assigned = 2,
        InProgress = 3,
        Completed = 4,
        Cancelled = 5
    }
}
