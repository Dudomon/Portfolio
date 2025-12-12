using Abp.Application.Services;
using Abp.Application.Services.Dto;
using WarehouseManagement.Inventory.Dto;

namespace WarehouseManagement.Inventory
{
    public interface IInventoryAppService : IAsyncCrudAppService<InventoryItemDto, long, PagedInventoryResultRequestDto, CreateInventoryItemDto, InventoryItemDto>
    {
        Task<List<InventoryItemDto>> GetItemsNeedingReorderAsync();
        Task<List<InventoryItemDto>> GetItemsByWarehouseAsync(long warehouseId);
        Task AdjustQuantityAsync(AdjustInventoryQuantityInput input);
        Task<InventoryItemDto> GetItemBySKUAsync(string sku);
    }

    public class PagedInventoryResultRequestDto : PagedAndSortedResultRequestDto
    {
        public long? WarehouseId { get; set; }
        public string? Keyword { get; set; }
        public bool? NeedsReorder { get; set; }
    }
}
