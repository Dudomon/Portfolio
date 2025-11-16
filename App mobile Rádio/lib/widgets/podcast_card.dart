import 'package:flutter/material.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:provider/provider.dart';
import '../models/podcast.dart';
import '../services/theme_service.dart';
import '../utils/theme.dart';

class PodcastCard extends StatelessWidget {
  final Podcast podcast;
  final VoidCallback? onPlay;
  
  const PodcastCard({
    super.key,
    required this.podcast,
    this.onPlay,
  });

  @override
  Widget build(BuildContext context) {
    return Consumer<ThemeService>(
      builder: (context, themeService, child) {
        return Card(
          margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
          child: InkWell(
            onTap: onPlay,
            borderRadius: BorderRadius.circular(12),
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Row(
                children: [
                  // Ícone do podcast
                  Container(
                    width: 50,
                    height: 50,
                    decoration: BoxDecoration(
                      color: AppTheme.primaryOrange,
                      borderRadius: BorderRadius.circular(25),
                      boxShadow: [
                        BoxShadow(
                          color: AppTheme.primaryOrange.withOpacity(0.3),
                          blurRadius: 8,
                          offset: const Offset(0, 4),
                        ),
                      ],
                    ),
                    child: ClipRRect(
                            borderRadius: BorderRadius.circular(25),
                            child: CachedNetworkImage(
                              imageUrl: podcast.imageUrl,
                              fit: BoxFit.cover,
                              placeholder: (context, url) => const Icon(
                                Icons.podcasts,
                                color: Colors.white,
                                size: 24,
                              ),
                              errorWidget: (context, url, error) => const Icon(
                                Icons.podcasts,
                                color: Colors.white,
                                size: 24,
                              ),
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
                          style: Theme.of(context).textTheme.titleMedium,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 4),
                        
                        // Data e duração
                        Row(
                          children: [
                            Icon(
                              Icons.access_time,
                              size: 12,
                              color: themeService.isDarkMode 
                                ? AppTheme.darkTextSecondary 
                                : AppTheme.lightTextSecondary,
                            ),
                            const SizedBox(width: 4),
                            Flexible(
                              child: Text(
                                podcast.formattedDate,
                                style: Theme.of(context).textTheme.labelSmall,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            const SizedBox(width: 8),
                            const Icon(
                              Icons.play_circle_outline,
                              size: 12,
                              color: AppTheme.primaryOrange,
                            ),
                            const SizedBox(width: 4),
                            Text(
                              podcast.duration,
                              style: Theme.of(context).textTheme.labelSmall?.copyWith(
                                color: AppTheme.primaryOrange,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  
                  // Botão play
                  Container(
                    width: 36,
                    height: 36,
                    decoration: BoxDecoration(
                      color: AppTheme.primaryOrange.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(
                        color: AppTheme.primaryOrange.withOpacity(0.3),
                        width: 1,
                      ),
                    ),
                    child: const Icon(
                      Icons.play_arrow,
                      color: AppTheme.primaryOrange,
                      size: 20,
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}