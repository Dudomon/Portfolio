using Abp;
using Abp.TestBase;
using WarehouseManagement.EntityFrameworkCore;

namespace WarehouseManagement.Tests
{
    public abstract class WarehouseManagementTestBase : AbpIntegratedTestBase<WarehouseManagementTestModule>
    {
        protected WarehouseManagementTestBase()
        {
            void NormalizeDbContext(WarehouseManagementDbContext context)
            {
                context.EntityChangeEventHelper = NullEntityChangeEventHelper.Instance;
                context.EventBus = NullEventBus.Instance;
                context.SuppressAutoSetTenantId = true;
            }

            UsingDbContext(context =>
            {
                NormalizeDbContext(context);
            });
        }

        protected void UsingDbContext(Action<WarehouseManagementDbContext> action)
        {
            UsingDbContext(AbpSession.TenantId, action);
        }

        protected Task UsingDbContextAsync(Func<WarehouseManagementDbContext, Task> action)
        {
            return UsingDbContextAsync(AbpSession.TenantId, action);
        }

        protected T UsingDbContext<T>(Func<WarehouseManagementDbContext, T> func)
        {
            return UsingDbContext(AbpSession.TenantId, func);
        }

        protected Task<T> UsingDbContextAsync<T>(Func<WarehouseManagementDbContext, Task<T>> func)
        {
            return UsingDbContextAsync(AbpSession.TenantId, func);
        }

        protected void UsingDbContext(int? tenantId, Action<WarehouseManagementDbContext> action)
        {
            using (UsingTenantId(tenantId))
            {
                using (var context = LocalIocManager.Resolve<WarehouseManagementDbContext>())
                {
                    action(context);
                    context.SaveChanges();
                }
            }
        }

        protected async Task UsingDbContextAsync(int? tenantId, Func<WarehouseManagementDbContext, Task> action)
        {
            using (UsingTenantId(tenantId))
            {
                using (var context = LocalIocManager.Resolve<WarehouseManagementDbContext>())
                {
                    await action(context);
                    await context.SaveChangesAsync();
                }
            }
        }

        protected T UsingDbContext<T>(int? tenantId, Func<WarehouseManagementDbContext, T> func)
        {
            T result;

            using (UsingTenantId(tenantId))
            {
                using (var context = LocalIocManager.Resolve<WarehouseManagementDbContext>())
                {
                    result = func(context);
                    context.SaveChanges();
                }
            }

            return result;
        }

        protected async Task<T> UsingDbContextAsync<T>(int? tenantId, Func<WarehouseManagementDbContext, Task<T>> func)
        {
            T result;

            using (UsingTenantId(tenantId))
            {
                using (var context = LocalIocManager.Resolve<WarehouseManagementDbContext>())
                {
                    result = await func(context);
                    await context.SaveChangesAsync();
                }
            }

            return result;
        }

        protected IDisposable UsingTenantId(int? tenantId)
        {
            var changeBack = false;
            var oldTenantId = AbpSession.TenantId;

            if (AbpSession.TenantId != tenantId)
            {
                AbpSession.TenantId = tenantId;
                changeBack = true;
            }

            return new DisposeAction(() =>
            {
                if (changeBack)
                {
                    AbpSession.TenantId = oldTenantId;
                }
            });
        }
    }

    public class WarehouseManagementTestModule : Abp.Modules.AbpModule
    {
    }
}
