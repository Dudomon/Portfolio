using Abp.BackgroundJobs;
using Abp.Dependency;
using Abp.Domain.Repositories;
using Abp.Domain.Uow;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using WarehouseManagement.Entities;

namespace WarehouseManagement.Jobs
{
    public class ReorderInventoryJob : BackgroundJob<ReorderInventoryJobArgs>, ITransientDependency
    {
        private readonly IRepository<InventoryItem, long> _inventoryRepository;
        private readonly ILogger<ReorderInventoryJob> _logger;

        public ReorderInventoryJob(
            IRepository<InventoryItem, long> inventoryRepository,
            ILogger<ReorderInventoryJob> logger)
        {
            _inventoryRepository = inventoryRepository;
            _logger = logger;
        }

        [UnitOfWork]
        public override void Execute(ReorderInventoryJobArgs args)
        {
            _logger.LogInformation("Starting ReorderInventoryJob for tenant: {TenantId}", args.TenantId);

            var itemsNeedingReorder = _inventoryRepository
                .GetAll()
                .Where(x => (x.QuantityOnHand - x.QuantityReserved) <= x.ReorderPoint && x.IsActive)
                .ToList();

            _logger.LogInformation("Found {Count} items needing reorder", itemsNeedingReorder.Count);

            foreach (var item in itemsNeedingReorder)
            {
                _logger.LogInformation(
                    "Item {SKU} needs reorder. Current: {Current}, Reorder Point: {ReorderPoint}",
                    item.SKU,
                    item.QuantityAvailable,
                    item.ReorderPoint);
            }

            _logger.LogInformation("ReorderInventoryJob completed");
        }
    }

    public class ReorderInventoryJobArgs
    {
        public int TenantId { get; set; }
    }
}
