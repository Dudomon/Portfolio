import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';
import '../services/news_service.dart';
import '../services/theme_service.dart';
import '../utils/theme.dart';
import 'news_card.dart';

class NewsList extends StatefulWidget {
  const NewsList({super.key});

  @override
  State<NewsList> createState() => _NewsListState();
}

class _NewsListState extends State<NewsList> {
  final PageController _pageController = PageController(viewportFraction: 0.85);
  int _currentPage = 0;

  @override
  void initState() {
    super.initState();
    _pageController.addListener(_onPageChanged);
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  void _onPageChanged() {
    final newsService = context.read<NewsService>();
    if (!_pageController.hasClients || newsService.news.isEmpty) return;
    
    final page = _pageController.page?.round() ?? 0;
    
    if (page != _currentPage && page >= 0 && page < newsService.news.length) {
      setState(() {
        _currentPage = page;
      });
      
      // Carrega mais notícias quando próximo do fim
      if (page >= newsService.news.length - 3) {
        newsService.loadMoreNews();
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Consumer2<NewsService, ThemeService>(
      builder: (context, newsService, themeService, child) {
        if (newsService.isLoading && newsService.news.isEmpty) {
          return const Center(
            child: CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation<Color>(AppTheme.primaryOrange),
            ),
          );
        }

        if (newsService.error != null && newsService.news.isEmpty) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  Icons.error_outline,
                  size: 64,
                  color: themeService.isDarkMode 
                    ? AppTheme.darkTextSecondary 
                    : AppTheme.lightTextSecondary,
                ),
                const SizedBox(height: 16),
                Text(
                  'Erro ao carregar notícias',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                Text(
                  newsService.error!,
                  style: Theme.of(context).textTheme.bodyMedium,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
                ElevatedButton(
                  onPressed: newsService.refreshNews,
                  child: const Text('Tentar novamente'),
                ),
              ],
            ),
          );
        }

        if (newsService.news.isEmpty) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  Icons.newspaper,
                  size: 64,
                  color: themeService.isDarkMode 
                    ? AppTheme.darkTextSecondary 
                    : AppTheme.lightTextSecondary,
                ),
                const SizedBox(height: 16),
                Text(
                  'Nenhuma notícia disponível',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
              ],
            ),
          );
        }

        return RefreshIndicator(
          onRefresh: newsService.refreshNews,
          color: AppTheme.primaryOrange,
          backgroundColor: AppTheme.darkCard,
          child: Column(
            children: [
              // Indicador de página e dots
              if (newsService.news.isNotEmpty)
                Container(
                  padding: const EdgeInsets.symmetric(vertical: 6),
                  child: Column(
                    children: [
                      // Contador simples
                      Text(
                        '${(_currentPage + 1).clamp(1, newsService.news.length)} de ${newsService.news.length}',
                        style: Theme.of(context).textTheme.labelMedium?.copyWith(
                          color: themeService.isDarkMode 
                            ? AppTheme.darkTextSecondary 
                            : AppTheme.lightTextSecondary,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      const SizedBox(height: 8),
                    
                    // Dots indicadores
                    if (newsService.news.isNotEmpty && newsService.news.length > 1)
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: List.generate(
                          newsService.news.length > 5 ? 5 : newsService.news.length,
                          (index) {
                            // Lógica para mostrar apenas 5 dots com indicação de posição
                            bool isActive = false;
                            if (newsService.news.length <= 5) {
                              isActive = index == _currentPage;
                            } else {
                              // Para mais de 5 notícias, mostra a posição relativa
                              if (newsService.news.isNotEmpty && _currentPage >= 0) {
                                final position = (_currentPage / newsService.news.length * 5).round().clamp(0, 4);
                                isActive = index == position;
                              } else {
                                isActive = index == 0;
                              }
                            }
                            
                            return AnimatedContainer(
                              duration: const Duration(milliseconds: 300),
                              margin: const EdgeInsets.symmetric(horizontal: 4),
                              width: isActive ? 20 : 8,
                              height: 8,
                              decoration: BoxDecoration(
                                color: isActive 
                                  ? AppTheme.primaryOrange 
                                  : (themeService.isDarkMode 
                                      ? AppTheme.darkTextSecondary.withOpacity(0.3)
                                      : AppTheme.lightTextSecondary.withOpacity(0.3)),
                                borderRadius: BorderRadius.circular(4),
                              ),
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                ),
              
              // Carrossel de notícias
              if (newsService.news.isNotEmpty)
                Expanded(
                  child: PageView.builder(
                    controller: _pageController,
                    padEnds: false,
                    itemCount: newsService.news.length,
                    itemBuilder: (context, index) {
                      // Efeito de escala baseado na proximidade da página atual
                      double scale = 1.0;
                      if (_pageController.hasClients && _pageController.page != null) {
                        final page = _pageController.page!;
                        if (page.isFinite && !page.isNaN) {
                          scale = 1.0 - (index - page).abs() * 0.1;
                          scale = scale.clamp(0.85, 1.0);
                        }
                      }
                      
                      return Transform.scale(
                        scale: scale,
                        child: Container(
                          margin: EdgeInsets.symmetric(
                            horizontal: MediaQuery.of(context).size.width * 0.02, 
                            vertical: 16
                          ),
                          child: NewsCard(news: newsService.news[index]),
                        ),
                      );
                    },
                  ),
                ),
              
              // Banner "Anuncie Aqui!"
              if (!newsService.isLoading) ...[
                GestureDetector(
                  onTap: () async {
                    final Uri whatsappUri = Uri.parse('https://wa.me/5549991686860');
                    if (await canLaunchUrl(whatsappUri)) {
                      await launchUrl(whatsappUri, mode: LaunchMode.externalApplication);
                    }
                  },
                  child: Container(
                    margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 16),
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          AppTheme.primaryOrange.withOpacity(0.15),
                          AppTheme.darkSurface.withOpacity(0.8),
                        ],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: AppTheme.primaryOrange.withOpacity(0.4),
                        width: 1,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: AppTheme.primaryOrange.withOpacity(0.2),
                          blurRadius: 12,
                          offset: const Offset(0, 6),
                        ),
                      ],
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        // Ícone de publicidade
                        Container(
                          width: 40,
                          height: 40,
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                AppTheme.primaryOrange,
                                AppTheme.primaryOrange.withOpacity(0.7),
                              ],
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                            ),
                            borderRadius: BorderRadius.circular(12),
                            boxShadow: [
                              BoxShadow(
                                color: AppTheme.primaryOrange.withOpacity(0.4),
                                blurRadius: 6,
                                offset: const Offset(0, 3),
                              ),
                            ],
                          ),
                          child: const Icon(
                            Icons.campaign_rounded,
                            color: Colors.white,
                            size: 22,
                          ),
                        ),
                        const SizedBox(width: 16),
                        // Texto "Anuncie Aqui!"
                        Text(
                          'Anuncie Aqui!',
                          style: Theme.of(context).textTheme.titleLarge?.copyWith(
                            fontWeight: FontWeight.bold,
                            letterSpacing: 0.5,
                            color: themeService.isDarkMode
                              ? Colors.white
                              : Colors.black,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
              
              // Loading indicator
              if (newsService.isLoading && newsService.news.isNotEmpty) ...[
                const Padding(
                  padding: EdgeInsets.all(16),
                  child: CircularProgressIndicator(
                    valueColor: AlwaysStoppedAnimation<Color>(
                      AppTheme.primaryOrange,
                    ),
                  ),
                ),
              ],
            ],
          ),
        );
      },
    );
  }
}