import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';
import '../services/podcast_service.dart';
import '../services/podcast_player_service.dart';
import '../services/theme_service.dart';
import '../utils/theme.dart';

class PodcastSection extends StatelessWidget {
  const PodcastSection({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer3<PodcastService, PodcastPlayerService, ThemeService>(
      builder: (context, podcastService, playerService, themeService, child) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            // Header
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Últimos Podcasts',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  TextButton(
                    onPressed: () => _openPodcastPage(),
                    child: const Text(
                      'Ver todos',
                      style: TextStyle(
                        color: AppTheme.primaryOrange,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            
            // Lista de podcasts
            Flexible(
              child: _buildPodcastList(podcastService, playerService, themeService, context),
            ),
          ],
        );
      },
    );
  }

  Widget _buildPodcastList(PodcastService podcastService, PodcastPlayerService playerService, ThemeService themeService, BuildContext context) {
    if (podcastService.isLoading) {
      return const Center(
        child: CircularProgressIndicator(
          valueColor: AlwaysStoppedAnimation<Color>(AppTheme.primaryOrange),
        ),
      );
    }

    if (podcastService.error != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.error_outline,
              size: 48,
              color: themeService.isDarkMode 
                ? AppTheme.darkTextSecondary 
                : AppTheme.lightTextSecondary,
            ),
            const SizedBox(height: 8),
            Text(
              'Erro ao carregar podcasts',
              style: TextStyle(
                color: themeService.isDarkMode 
                  ? AppTheme.darkTextSecondary 
                  : AppTheme.lightTextSecondary,
              ),
            ),
          ],
        ),
      );
    }

    if (podcastService.latestPodcasts.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.podcasts,
              size: 48,
              color: themeService.isDarkMode 
                ? AppTheme.darkTextSecondary 
                : AppTheme.lightTextSecondary,
            ),
            const SizedBox(height: 8),
            Text(
              'Nenhum podcast disponível',
              style: TextStyle(
                color: themeService.isDarkMode 
                  ? AppTheme.darkTextSecondary 
                  : AppTheme.lightTextSecondary,
              ),
            ),
          ],
        ),
      );
    }

    // Lista vertical de podcasts (máximo 5 para melhor visualização)
    return Column(
      children: podcastService.latestPodcasts.take(5).map((podcast) {
        return Container(
          margin: EdgeInsets.symmetric(
            horizontal: MediaQuery.of(context).size.width * 0.04, 
            vertical: 3
          ),
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: themeService.isDarkMode 
              ? AppTheme.darkCard 
              : AppTheme.lightCard,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: AppTheme.primaryOrange.withOpacity(0.2),
              width: 1,
            ),
            boxShadow: [
              BoxShadow(
                color: themeService.isDarkMode 
                  ? AppTheme.primaryOrange.withOpacity(0.1)
                  : Colors.black.withOpacity(0.08),
                blurRadius: 6,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: InkWell(
            onTap: () {
              playerService.playPodcast(podcast);
            },
            borderRadius: BorderRadius.circular(12),
            child: Row(
              children: [
                // Imagem do podcast
                Container(
                  width: 60,
                  height: 60,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(8),
                    color: AppTheme.primaryOrange.withOpacity(0.1),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: podcast.imageUrl.isNotEmpty 
                      ? Image.network(
                          podcast.imageUrl,
                          fit: BoxFit.cover,
                          errorBuilder: (context, error, stackTrace) {
                            return Icon(
                              Icons.podcasts,
                              color: AppTheme.primaryOrange.withOpacity(0.5),
                              size: 30,
                            );
                          },
                        )
                      : Icon(
                          Icons.podcasts,
                          color: AppTheme.primaryOrange.withOpacity(0.5),
                          size: 30,
                        ),
                  ),
                ),
                const SizedBox(width: 12),
                
                // Informações do podcast
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Título
                      Text(
                        podcast.title,
                        style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w600,
                          height: 1.2,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 4),
                      
                      // Data e duração
                      Row(
                        children: [
                          Icon(
                            Icons.access_time_rounded,
                            size: 14,
                            color: themeService.isDarkMode 
                              ? AppTheme.darkTextSecondary 
                              : AppTheme.lightTextSecondary,
                          ),
                          const SizedBox(width: 4),
                          Flexible(
                            child: Text(
                              podcast.formattedDate,
                              style: Theme.of(context).textTheme.labelSmall?.copyWith(
                                color: themeService.isDarkMode 
                                  ? AppTheme.darkTextSecondary 
                                  : AppTheme.lightTextSecondary,
                              ),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          const SizedBox(width: 8),
                          const Icon(
                            Icons.play_circle_outline_rounded,
                            size: 14,
                            color: AppTheme.primaryOrange,
                          ),
                          const SizedBox(width: 4),
                          Flexible(
                            child: Text(
                              podcast.duration,
                              style: Theme.of(context).textTheme.labelSmall?.copyWith(
                                color: AppTheme.primaryOrange,
                                fontWeight: FontWeight.w500,
                              ),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                
                // Botão de play
                Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: AppTheme.primaryOrange.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Icon(
                    playerService.currentPodcast?.id == podcast.id && playerService.isPlaying
                      ? Icons.pause_rounded 
                      : Icons.play_arrow_rounded,
                    color: AppTheme.primaryOrange,
                    size: 24,
                  ),
                ),
              ],
            ),
          ),
        );
      }).toList(),
    );
  }

  Future<void> _openPodcastPage() async {
    final url = Uri.parse('https://radioentrerios.com.br/podcasts/');
    if (await canLaunchUrl(url)) {
      await launchUrl(
        url,
        mode: LaunchMode.externalApplication,
      );
    }
  }
}