// ARVTON — Screen 3: GarmentPickerScreen
// 2-column grid of garments with category tabs, selection state,
// floating "Try On" bar when garment is selected.

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:go_router/go_router.dart';
import 'package:path_provider/path_provider.dart';
import 'package:dio/dio.dart';
import '../providers/tryon_provider.dart';

/// Garment catalog item
class GarmentItem {
  final String id;
  final String name;
  final String price;
  final String category;
  final String imageUrl;

  const GarmentItem({
    required this.id,
    required this.name,
    required this.price,
    required this.category,
    required this.imageUrl,
  });
}

/// Hardcoded catalog with 16+ items across 4 categories
const List<GarmentItem> _garmentCatalog = [
  // T-Shirts
  GarmentItem(
    id: 'ts01',
    name: 'Classic White Tee',
    price: '\$29',
    category: 'T-Shirts',
    imageUrl: 'https://picsum.photos/seed/tshirt1/400/500',
  ),
  GarmentItem(
    id: 'ts02',
    name: 'Midnight Black Tee',
    price: '\$29',
    category: 'T-Shirts',
    imageUrl: 'https://picsum.photos/seed/tshirt2/400/500',
  ),
  GarmentItem(
    id: 'ts03',
    name: 'Ocean Blue Tee',
    price: '\$32',
    category: 'T-Shirts',
    imageUrl: 'https://picsum.photos/seed/tshirt3/400/500',
  ),
  GarmentItem(
    id: 'ts04',
    name: 'Sunset Orange Tee',
    price: '\$32',
    category: 'T-Shirts',
    imageUrl: 'https://picsum.photos/seed/tshirt4/400/500',
  ),
  // Shirts
  GarmentItem(
    id: 'sh01',
    name: 'Oxford Button Down',
    price: '\$59',
    category: 'Shirts',
    imageUrl: 'https://picsum.photos/seed/shirt1/400/500',
  ),
  GarmentItem(
    id: 'sh02',
    name: 'Linen Chambray',
    price: '\$65',
    category: 'Shirts',
    imageUrl: 'https://picsum.photos/seed/shirt2/400/500',
  ),
  GarmentItem(
    id: 'sh03',
    name: 'Flannel Plaid',
    price: '\$55',
    category: 'Shirts',
    imageUrl: 'https://picsum.photos/seed/shirt3/400/500',
  ),
  GarmentItem(
    id: 'sh04',
    name: 'Silk Mandarin',
    price: '\$89',
    category: 'Shirts',
    imageUrl: 'https://picsum.photos/seed/shirt4/400/500',
  ),
  // Dresses
  GarmentItem(
    id: 'dr01',
    name: 'Summer Floral Midi',
    price: '\$79',
    category: 'Dresses',
    imageUrl: 'https://picsum.photos/seed/dress1/400/500',
  ),
  GarmentItem(
    id: 'dr02',
    name: 'Little Black Dress',
    price: '\$89',
    category: 'Dresses',
    imageUrl: 'https://picsum.photos/seed/dress2/400/500',
  ),
  GarmentItem(
    id: 'dr03',
    name: 'Wrap Emerald',
    price: '\$75',
    category: 'Dresses',
    imageUrl: 'https://picsum.photos/seed/dress3/400/500',
  ),
  GarmentItem(
    id: 'dr04',
    name: 'Satin Evening Gown',
    price: '\$129',
    category: 'Dresses',
    imageUrl: 'https://picsum.photos/seed/dress4/400/500',
  ),
  // Trousers
  GarmentItem(
    id: 'tr01',
    name: 'Slim Chinos',
    price: '\$49',
    category: 'Trousers',
    imageUrl: 'https://picsum.photos/seed/trouser1/400/500',
  ),
  GarmentItem(
    id: 'tr02',
    name: 'Wide-Leg Linen',
    price: '\$55',
    category: 'Trousers',
    imageUrl: 'https://picsum.photos/seed/trouser2/400/500',
  ),
  GarmentItem(
    id: 'tr03',
    name: 'Tailored Wool',
    price: '\$85',
    category: 'Trousers',
    imageUrl: 'https://picsum.photos/seed/trouser3/400/500',
  ),
  GarmentItem(
    id: 'tr04',
    name: 'Cargo Joggers',
    price: '\$45',
    category: 'Trousers',
    imageUrl: 'https://picsum.photos/seed/trouser4/400/500',
  ),
];

final _categories = ['All', 'T-Shirts', 'Shirts', 'Dresses', 'Trousers'];

class GarmentPickerScreen extends ConsumerStatefulWidget {
  const GarmentPickerScreen({super.key});

  @override
  ConsumerState<GarmentPickerScreen> createState() =>
      _GarmentPickerScreenState();
}

class _GarmentPickerScreenState extends ConsumerState<GarmentPickerScreen> {
  String _selectedCategory = 'All';
  String? _selectedGarmentId;
  bool _isSubmitting = false;

  List<GarmentItem> get _filteredGarments {
    if (_selectedCategory == 'All') return _garmentCatalog;
    return _garmentCatalog
        .where((g) => g.category == _selectedCategory)
        .toList();
  }

  GarmentItem? get _selectedGarment {
    if (_selectedGarmentId == null) return null;
    return _garmentCatalog.firstWhere((g) => g.id == _selectedGarmentId);
  }

  Future<void> _submitTryon() async {
    final garment = _selectedGarment;
    if (garment == null) return;

    setState(() => _isSubmitting = true);

    try {
      // Download garment image to a temp file for the API
      final tempDir = await getTemporaryDirectory();
      final tempFile = File('${tempDir.path}/garment_${garment.id}.jpg');

      if (!tempFile.existsSync()) {
        final dio = Dio();
        await dio.download(garment.imageUrl, tempFile.path);
      }

      ref.read(tryonProvider.notifier).selectGarment(garment.id, tempFile);
      await ref.read(tryonProvider.notifier).submitTryon(quality: 'auto');

      if (mounted) {
        context.push('/processing');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: $e'),
            backgroundColor: Colors.red.shade700,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isSubmitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final tryonState = ref.watch(tryonProvider);

    return Scaffold(
      backgroundColor: const Color(0xFF0D0D2B),
      body: SafeArea(
        child: Column(
          children: [
            // Header with person thumbnail
            _buildHeader(tryonState),

            // Category tabs
            _buildCategoryTabs(),

            // Garment grid
            Expanded(child: _buildGarmentGrid()),

            // Floating "Try On" bar
            if (_selectedGarmentId != null) _buildTryOnBar(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(TryonState state) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      child: Row(
        children: [
          // Back button
          GestureDetector(
            onTap: () => context.pop(),
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.08),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Icon(
                Icons.arrow_back_ios_new,
                color: Colors.white,
                size: 18,
              ),
            ),
          ),

          const SizedBox(width: 12),

          // Person thumbnail
          if (state.personImage != null)
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: const Color(0xFF6C63FF), width: 2),
                image: DecorationImage(
                  image: FileImage(state.personImage!),
                  fit: BoxFit.cover,
                ),
              ),
            ),

          const SizedBox(width: 12),

          // Title
          const Expanded(
            child: Text(
              'Choose a Garment',
              style: TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCategoryTabs() {
    return Container(
      height: 44,
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        itemCount: _categories.length,
        itemBuilder: (context, index) {
          final cat = _categories[index];
          final isSelected = cat == _selectedCategory;

          return GestureDetector(
            onTap: () => setState(() => _selectedCategory = cat),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 200),
              margin: const EdgeInsets.only(right: 8),
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
              decoration: BoxDecoration(
                color: isSelected
                    ? const Color(0xFF6C63FF)
                    : Colors.white.withOpacity(0.06),
                borderRadius: BorderRadius.circular(22),
                border: Border.all(
                  color: isSelected
                      ? const Color(0xFF6C63FF)
                      : Colors.white.withOpacity(0.1),
                ),
              ),
              child: Text(
                cat,
                style: TextStyle(
                  color: isSelected ? Colors.white : Colors.white70,
                  fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
                  fontSize: 13,
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildGarmentGrid() {
    final garments = _filteredGarments;

    return GridView.builder(
      padding: const EdgeInsets.all(16),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        childAspectRatio: 0.7,
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
      ),
      itemCount: garments.length,
      itemBuilder: (context, index) {
        final garment = garments[index];
        final isSelected = garment.id == _selectedGarmentId;

        return GestureDetector(
          onTap: () {
            setState(() {
              _selectedGarmentId = _selectedGarmentId == garment.id
                  ? null
                  : garment.id;
            });
          },
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.05),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: isSelected
                    ? const Color(0xFF6C63FF)
                    : Colors.white.withOpacity(0.08),
                width: isSelected ? 3 : 1,
              ),
            ),
            child: Stack(
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Image
                    Expanded(
                      child: ClipRRect(
                        borderRadius: const BorderRadius.vertical(
                          top: Radius.circular(14),
                        ),
                        child: CachedNetworkImage(
                          imageUrl: garment.imageUrl,
                          fit: BoxFit.cover,
                          placeholder: (_, __) => Container(
                            color: Colors.white.withOpacity(0.05),
                            child: const Center(
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: Color(0xFF6C63FF),
                              ),
                            ),
                          ),
                          errorWidget: (_, __, ___) => Container(
                            color: Colors.white.withOpacity(0.05),
                            child: const Icon(
                              Icons.broken_image,
                              color: Colors.white30,
                            ),
                          ),
                        ),
                      ),
                    ),

                    // Name + price
                    Padding(
                      padding: const EdgeInsets.all(10),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            garment.name,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 13,
                              fontWeight: FontWeight.w500,
                            ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                          const SizedBox(height: 4),
                          Text(
                            garment.price,
                            style: const TextStyle(
                              color: Color(0xFF6C63FF),
                              fontSize: 14,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),

                // Checkmark overlay
                if (isSelected)
                  Positioned(
                    top: 8,
                    right: 8,
                    child: Container(
                      padding: const EdgeInsets.all(4),
                      decoration: const BoxDecoration(
                        color: Color(0xFF6C63FF),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.check,
                        color: Colors.white,
                        size: 16,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildTryOnBar() {
    final garment = _selectedGarment;
    if (garment == null) return const SizedBox.shrink();

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1A40),
        border: Border(top: BorderSide(color: Colors.white.withOpacity(0.08))),
      ),
      child: SafeArea(
        top: false,
        child: Row(
          children: [
            // Selected garment thumbnail
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: CachedNetworkImage(
                imageUrl: garment.imageUrl,
                width: 52,
                height: 52,
                fit: BoxFit.cover,
              ),
            ),

            const SizedBox(width: 12),

            // Garment info
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    garment.name,
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                      fontSize: 14,
                    ),
                  ),
                  Text(
                    garment.price,
                    style: const TextStyle(
                      color: Color(0xFF6C63FF),
                      fontWeight: FontWeight.w700,
                      fontSize: 13,
                    ),
                  ),
                ],
              ),
            ),

            // Try On button
            ElevatedButton(
              onPressed: _isSubmitting ? null : _submitTryon,
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF6C63FF),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(
                  horizontal: 28,
                  vertical: 14,
                ),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
                elevation: 8,
                shadowColor: const Color(0xFF6C63FF).withOpacity(0.4),
              ),
              child: _isSubmitting
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.checkroom, size: 18),
                        SizedBox(width: 6),
                        Text(
                          'Try On',
                          style: TextStyle(fontWeight: FontWeight.w700),
                        ),
                      ],
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
