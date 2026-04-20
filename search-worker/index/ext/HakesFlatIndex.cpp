#include "search-worker/index/ext/HakesFilterIndex.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <cmath>

#include "search-worker/index/ext/index_io_ext.h"
#include "search-worker/index/ext/utils.h"
#include "utils/fileutil.h"

#define FILTER_RINDEX_NAME "rindex.bin"
#define FILTER_INDEX_NAME "filter_index.bin"

namespace faiss {

inline float L2Sqr(const float* a, const float* b, int d) {
  float sum = 0.0f;
  for (int i = 0; i < d; i++) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

HakesFilterIndex::HakesFilterIndex() {
  del_checker_.reset(new TagChecker<idx_t>());
}

HakesFilterIndex::~HakesFilterIndex() {
  for (size_t i = 0; i < link_lists_.size(); i++) {
    if (link_lists_[i] != nullptr) {
      free(link_lists_[i]);
    }
  }
}

bool HakesFilterIndex::Initialize(const std::string& path, int mode,
                                  bool keep_pa) {
  std::string rindex_path = path + "/" + FILTER_RINDEX_NAME;
  std::unique_ptr<hakes::FileIOReader> rf =
      hakes::IsFileExist(rindex_path)
          ? std::unique_ptr<hakes::FileIOReader>(
                new hakes::FileIOReader(rindex_path.c_str()))
          : nullptr;
  
  bool loaded = load_hakes_filterindex(rf.get(), this);
  
  std::string filter_path = path + "/" + FILTER_INDEX_NAME;
  if (hakes::IsFileExist(filter_path)) {
    if (!LoadFilterIndex(filter_path)) {
      fprintf(stderr, "Failed to load filter index from %s\n", filter_path.c_str());
    }
  }
  
  return loaded;
}

bool HakesFilterIndex::AddWithIds(int n, int d, const float* vecs,
                                  const faiss::idx_t* ids, faiss::idx_t* assign,
                                  int* vecs_t_d,
                                  std::unique_ptr<float[]>* vecs_t) {
  if (ntotal_ == 0) {
    d_ = d;
    if (max_elements_ == 0) {
      size_t initial_capacity = std::max(static_cast<size_t>(n * 2), static_cast<size_t>(1024));
      InitFavorParams(initial_capacity, 16, 200, 100);
    }
  }
  
  if (!BuildFilterIndex(n, d, vecs, ids)) {
    fprintf(stderr, "Failed to build filter index\n");
    return false;
  }
  
  if (refine_index_) {
    refine_index_->add_with_ids(n, vecs, ids);
  }
  
  ntotal_ += n;
  return true;
}

bool HakesFilterIndex::Search(int n, int d, const float* query,
                              const HakesSearchParams& params,
                              std::unique_ptr<float[]>* distances,
                              std::unique_ptr<faiss::idx_t[]>* labels) {
  if (!distances || !labels) {
    throw std::runtime_error("distances and labels must not be nullptr");
  }

  if (n <= 0 || params.k <= 0) {
    return false;
  }

  distances->reset(new float[n * params.k]);
  labels->reset(new idx_t[n * params.k]);

  if (cur_element_count_ > 0) {
    return SearchWithFilter(n, d, query, params.k, params.id_selector,
                            distances->get(), labels->get());
  }

  if (params.id_selector) {
    faiss::SearchParameters search_params;
    search_params.sel = const_cast<faiss::IDSelector*>(params.id_selector);
    refine_index_->search(n, query, params.k, distances->get(), labels->get(),
                          &search_params);
  } else {
    refine_index_->search(n, query, params.k, distances->get(), labels->get());
  }

  return true;
}

bool HakesFilterIndex::Rerank(int n, int d, const float* query, int k,
                              faiss::idx_t* k_base_count,
                              faiss::idx_t* base_labels, float* base_distances,
                              std::unique_ptr<float[]>* distances,
                              std::unique_ptr<faiss::idx_t[]>* labels) {
  if (!base_labels || !base_distances) {
    throw std::runtime_error(
        "base distances and base labels must not be nullptr");
  }

  if (!distances || !labels) {
    throw std::runtime_error("distances and labels must not be nullptr");
  }

  faiss::idx_t total_labels = 0;
  std::vector<faiss::idx_t> base_label_start(n);
  for (int i = 0; i < n; i++) {
    base_label_start[i] = total_labels;
    total_labels += k_base_count[i];
  }

  std::unique_ptr<float[]> dist(new float[n * k]);
  std::memset(dist.get(), 0, n * k * sizeof(float));
  std::unique_ptr<faiss::idx_t[]> lab(new faiss::idx_t[n * k]);
  std::memset(lab.get(), 0, n * k * sizeof(faiss::idx_t));

  for (int i = 0; i < n; i++) {
    refine_index_->compute_distance_subset(
        1, query + i * d, k_base_count[i], base_distances + base_label_start[i],
        base_labels + base_label_start[i]);

    if (k >= k_base_count[i]) {
      std::memcpy(lab.get() + k * i, base_labels + base_label_start[i],
                  k_base_count[i] * sizeof(faiss::idx_t));
      std::memcpy(dist.get() + k * i, base_distances + base_label_start[i],
                  k_base_count[i] * sizeof(float));
      for (int j = k_base_count[i]; j < k; j++) {
        (lab.get() + k * i)[j] = -1;
      }
    }

    if (refine_index_->metric_type == faiss::METRIC_L2) {
      typedef faiss::CMax<float, faiss::idx_t> C;
      faiss::reorder_2_heaps<C>(1, (k > k_base_count[i]) ? k_base_count[i] : k,
                                lab.get() + k * i, dist.get() + k * i,
                                k_base_count[i],
                                base_labels + base_label_start[i],
                                base_distances + base_label_start[i]);
    } else if (refine_index_->metric_type == faiss::METRIC_INNER_PRODUCT) {
      typedef faiss::CMin<float, faiss::idx_t> C;
      faiss::reorder_2_heaps<C>(1, (k > k_base_count[i]) ? k_base_count[i] : k,
                                lab.get() + k * i, dist.get() + k * i,
                                k_base_count[i],
                                base_labels + base_label_start[i],
                                base_distances + base_label_start[i]);
    } else {
      throw std::runtime_error("Metric type not supported");
    }
  }

  distances->reset(dist.release());
  labels->reset(lab.release());
  return true;
}

bool HakesFilterIndex::Checkpoint(const std::string& checkpoint_path) const {
  std::string rindex_path = checkpoint_path + "/" + FILTER_RINDEX_NAME;
  std::unique_ptr<hakes::FileIOWriter> rf =
      std::unique_ptr<hakes::FileIOWriter>(
          new hakes::FileIOWriter(rindex_path.c_str()));
  save_hakes_filterindex(rf.get(), this);
  
  if (cur_element_count_ > 0) {
    std::string filter_path = checkpoint_path + "/" + FILTER_INDEX_NAME;
    if (!SaveFilterIndex(filter_path)) {
      fprintf(stderr, "Failed to save filter index to %s\n", filter_path.c_str());
      return false;
    }
  }
  
  return true;
}

std::string HakesFilterIndex::to_string() const {
  std::string ret;
  if (cur_element_count_ > 0) {
    ret = "HakesFilterIndex (FAVOR): n " + std::to_string(cur_element_count_) + 
          ", d " + std::to_string(d_) + 
          ", metric: " + (metric_type_ == METRIC_INNER_PRODUCT ? "ip" : "l2") +
          ", M: " + std::to_string(M_) +
          ", ef_construction: " + std::to_string(ef_construction_);
  } else if (refine_index_) {
    ret = "HakesFilterIndex (fallback): refine_index n " +
          std::to_string(refine_index_->ntotal) + ", d " +
          std::to_string(refine_index_->d) + ", metric: " +
          (refine_index_->metric_type == faiss::METRIC_INNER_PRODUCT ? "ip"
                                                                      : "l2");
  } else {
    ret = "HakesFilterIndex: no index loaded";
  }
  return ret;
}

void HakesFilterIndex::UpdateIndex(const HakesCollection* other) {
  const HakesFilterIndex* other_idx = dynamic_cast<const HakesFilterIndex*>(other);
  if (!other_idx) {
    fprintf(stderr, "UpdateIndex: incompatible index type\n");
    return;
  }
  
  if (cur_element_count_ == 0) {
    fprintf(stderr, "UpdateIndex: filter index not initialized\n");
    return;
  }
  
  fprintf(stderr, "UpdateIndex: updating filter index parameters\n");
  
  delta_d_ = other_idx->delta_d_;
  ef_ = other_idx->ef_;
}

bool HakesFilterIndex::AddBase(int n, int d, const float* vecs,
                               const faiss::idx_t* ids) {
  return BuildFilterIndex(n, d, vecs, ids);
}

std::string HakesFilterIndex::GetParams() const {
  return "M=" + std::to_string(M_) + 
         ",ef_construction=" + std::to_string(ef_construction_) +
         ",ef=" + std::to_string(ef_) +
         ",delta_d=" + std::to_string(delta_d_);
}

void HakesFilterIndex::InitFavorParams(size_t max_elements, size_t M, 
                                       size_t ef_construction, size_t random_seed) {
  max_elements_ = max_elements;
  M_ = M;
  maxM_ = M_;
  maxM0_ = M_ * 2;
  ef_construction_ = std::max(ef_construction, M_);
  ef_ = 100;
  
  level_generator_.seed(random_seed);
  update_probability_generator_.seed(random_seed + 1);
  
  size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
  size_data_per_element_ = size_links_level0_ + d_ * sizeof(float) + sizeof(idx_t);
  offset_data_ = size_links_level0_;
  label_offset_ = size_links_level0_ + d_ * sizeof(float);
  offset_level0_ = 0;
  
  data_level0_memory_.resize(max_elements_ * size_data_per_element_);
  
  cur_element_count_ = 0;
  
  element_levels_.resize(max_elements_, 0);
  link_lists_.resize(max_elements_, nullptr);
  link_list_locks_.clear();
  link_list_locks_.reserve(max_elements_);
  for (size_t i = 0; i < max_elements_; i++) {
    link_list_locks_.emplace_back(std::make_unique<std::mutex>());
  }
  
  mult_ = 1.0f / std::log(1.0f * static_cast<float>(M_));
  enterpoint_node_ = 0;
  maxlevel_ = -1;
  
  delta_d_ = 0.0f;
}

int GetRandomLevel(std::mt19937& level_generator, float mult) {
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  double r = -std::log(distribution(level_generator)) * mult;
  return static_cast<int>(r);
}

bool HakesFilterIndex::BuildFilterIndex(int n, int d, const float* vecs, 
                                        const idx_t* ids) {
  if (d_ == 0) {
    d_ = d;
  }
  
  if (cur_element_count_ + static_cast<size_t>(n) > max_elements_) {
    size_t new_capacity = std::max(max_elements_ * 2, cur_element_count_ + n + 1024);
    
    data_level0_memory_.resize(new_capacity * size_data_per_element_);
    element_levels_.resize(new_capacity, 0);
    link_lists_.resize(new_capacity, nullptr);
    link_list_locks_.reserve(new_capacity);
    while (link_list_locks_.size() < new_capacity) {
      link_list_locks_.emplace_back(std::make_unique<std::mutex>());
    }
    
    max_elements_ = new_capacity;
  }
  
  for (int i = 0; i < n; i++) {
    const float* vec = vecs + i * d;
    idx_t label = ids[i];
    
    {
      std::unique_lock<std::mutex> lock(label_lookup_lock_);
      auto it = label_lookup_.find(label);
      if (it != label_lookup_.end()) {
        tableint internal_id = it->second;
        std::memcpy(&data_level0_memory_[internal_id * size_data_per_element_ + offset_data_],
                    vec, d * sizeof(float));
        continue;
      }
      
      if (cur_element_count_ >= max_elements_) {
        throw std::runtime_error("The number of elements exceeds the specified limit");
      }
    }
    
    tableint cur_c = cur_element_count_++;
    
    {
      std::unique_lock<std::mutex> lock(label_lookup_lock_);
      label_lookup_[label] = cur_c;
    }
    
    int curlevel = GetRandomLevel(level_generator_, mult_);
    element_levels_[cur_c] = curlevel;
    
    std::memset(&data_level0_memory_[cur_c * size_data_per_element_ + offset_level0_],
                0, size_data_per_element_);
    
    std::memcpy(&data_level0_memory_[cur_c * size_data_per_element_ + label_offset_],
                &label, sizeof(idx_t));
    std::memcpy(&data_level0_memory_[cur_c * size_data_per_element_ + offset_data_],
                vec, d * sizeof(float));
    
    if (curlevel > 0) {
      size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
      link_lists_[cur_c] = static_cast<char*>(malloc(size_links_per_element_ * curlevel));
      if (link_lists_[cur_c] == nullptr) {
        throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
      }
      std::memset(link_lists_[cur_c], 0, size_links_per_element_ * curlevel);
    }
    
    float local_delta_d = 0.0f;
    
    if (static_cast<signed>(cur_c) > 0) {
      tableint currObj = enterpoint_node_;
      
      if (curlevel < maxlevel_) {
        float curdist = L2Sqr(vec, GetDataByInternalId(currObj), d_);
        
        for (int lvl = maxlevel_; lvl > curlevel; lvl--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int* data = reinterpret_cast<unsigned int*>(
                link_lists_[currObj] + (lvl - 1) * size_links_per_element_);
            int size = *reinterpret_cast<linklistsizeint*>(data);
            tableint* datal = reinterpret_cast<tableint*>(data + 1);
            
            for (int j = 0; j < size; j++) {
              tableint cand = datal[j];
              float d_cand = L2Sqr(vec, GetDataByInternalId(cand), d_);
              if (d_cand < curdist) {
                curdist = d_cand;
                currObj = cand;
                changed = true;
              }
            }
          }
        }
      }
      
      for (int lvl = std::min(curlevel, maxlevel_); lvl >= 0; lvl--) {
        std::priority_queue<std::pair<float, tableint>,
                            std::vector<std::pair<float, tableint>>,
                            FavorCompareByFirst<float>> top_candidates;
        SearchBaseLayerFilter(currObj, vec, lvl, top_candidates);
        
        if (lvl == 0 && top_candidates.size() == ef_construction_) {
          float rate = static_cast<float>(ef_construction_ - 10);
          auto temp = top_candidates;
          std::vector<std::pair<float, tableint>> elements;
          while (!temp.empty()) {
            elements.push_back(temp.top());
            temp.pop();
          }
          
          float diff = elements[0].first - elements[elements.size() - 10].first;
          local_delta_d += 5.0f * diff / (rate * static_cast<float>(max_elements_));
        }
        
        currObj = MutuallyConnectNewElement(vec, cur_c, top_candidates, lvl);
      }
    } else {
      enterpoint_node_ = 0;
      maxlevel_ = curlevel;
    }
    
    {
      std::lock_guard<std::mutex> lock(delta_mutex_);
      delta_d_ += local_delta_d;
    }
    
    if (curlevel > maxlevel_) {
      enterpoint_node_ = cur_c;
      maxlevel_ = curlevel;
    }
  }
  
  return true;
}

bool HakesFilterIndex::SearchWithFilter(int n, int d, const float* query, 
                                        int k, const faiss::IDSelector* id_selector,
                                        float* distances, idx_t* labels) {
  if (cur_element_count_ == 0) {
    return false;
  }
  
  for (int q = 0; q < n; q++) {
    const float* qvec = query + q * d;
    
    float selectivity = 1.0f;
    if (id_selector) {
      selectivity = SelectivityEstimator(id_selector);
    }
    
    std::vector<std::pair<float, idx_t>> results;
    
    if (selectivity > 0.01f) {
      SearchGraphFilter(qvec, k, selectivity, id_selector, results);
      if (results.size() < static_cast<size_t>(k) && id_selector != nullptr) {
        SearchBruteForceFilter(qvec, k, id_selector, results);
      }
    } else {
      SearchBruteForceFilter(qvec, k, id_selector, results);
    }
    
    size_t result_count = std::min(results.size(), static_cast<size_t>(k));
    for (size_t i = 0; i < result_count; i++) {
      distances[q * k + i] = results[i].first;
      labels[q * k + i] = results[i].second;
    }
    
    for (size_t i = result_count; i < static_cast<size_t>(k); i++) {
      distances[q * k + i] = LARGE_DIST;
      labels[q * k + i] = -1;
    }
  }
  
  return true;
}

float HakesFilterIndex::SelectivityEstimator(const faiss::IDSelector* id_selector) const {
  if (cur_element_count_ == 0 || id_selector == nullptr) {
    return 1.0f;
  }
  
  size_t count = 0;
  size_t sample_size = std::min(cur_element_count_, static_cast<size_t>(1000));
  size_t step = cur_element_count_ / sample_size;
  if (step == 0) step = 1;
  
  for (size_t i = 0; i < cur_element_count_; i += step) {
    idx_t label = GetExternalLabel(static_cast<tableint>(i));
    if (id_selector->is_member(label)) {
      count++;
    }
  }
  
  return static_cast<float>(count) / static_cast<float>(sample_size);
}

void HakesFilterIndex::MarkDeleted(idx_t label) {
  std::unique_lock<std::mutex> lock(label_lookup_lock_);
  auto it = label_lookup_.find(label);
  if (it != label_lookup_.end()) {
    std::unique_lock<std::mutex> del_lock(deleted_elements_lock_);
    deleted_elements_.insert(it->second);
  }
}

bool HakesFilterIndex::IsMarkedDeleted(tableint internal_id) const {
  std::unique_lock<std::mutex> lock(deleted_elements_lock_);
  return deleted_elements_.find(internal_id) != deleted_elements_.end();
}

tableint HakesFilterIndex::GetInternalId(idx_t label) const {
  std::unique_lock<std::mutex> lock(label_lookup_lock_);
  auto it = label_lookup_.find(label);
  if (it != label_lookup_.end()) {
    return it->second;
  }
  return static_cast<tableint>(-1);
}

idx_t HakesFilterIndex::GetExternalLabel(tableint internal_id) const {
  if (internal_id >= cur_element_count_) {
    return -1;
  }
  idx_t label;
  std::memcpy(&label, 
              &data_level0_memory_[internal_id * size_data_per_element_ + label_offset_],
              sizeof(idx_t));
  return label;
}

const float* HakesFilterIndex::GetDataByInternalId(tableint internal_id) const {
  return reinterpret_cast<const float*>(
      &data_level0_memory_[internal_id * size_data_per_element_ + offset_data_]);
}

void HakesFilterIndex::GetNeighborsByHeuristic2(
    std::priority_queue<std::pair<float, tableint>,
                        std::vector<std::pair<float, tableint>>,
                        FavorCompareByFirst<float>>& top_candidates,
    size_t M) {
  if (top_candidates.size() < M) {
    return;
  }

  std::vector<std::pair<float, tableint>> candidates;
  candidates.reserve(top_candidates.size());
  while (!top_candidates.empty()) {
    candidates.push_back(top_candidates.top());
    top_candidates.pop();
  }

  std::reverse(candidates.begin(), candidates.end());
  std::vector<std::pair<float, tableint>> return_list;
  return_list.reserve(M);

  for (const auto& current_pair : candidates) {
    if (return_list.size() >= M)
      break;

    bool good = true;
    float dist_to_query = current_pair.first;

    for (const auto& second_pair : return_list) {
      float curdist = L2Sqr(GetDataByInternalId(second_pair.second),
                            GetDataByInternalId(current_pair.second), d_);
      if (curdist < dist_to_query) {
        good = false;
        break;
      }
    }
    if (good) {
      return_list.push_back(current_pair);
    }
  }

  for (const auto& current_pair : return_list) {
    top_candidates.emplace(current_pair.first, current_pair.second);
  }
}

tableint HakesFilterIndex::MutuallyConnectNewElement(
    const float* data_point, tableint cur_c,
    std::priority_queue<std::pair<float, tableint>,
                        std::vector<std::pair<float, tableint>>,
                        FavorCompareByFirst<float>>& top_candidates,
    int level) {
  size_t Mcurmax = level ? maxM_ : maxM0_;
  GetNeighborsByHeuristic2(top_candidates, M_);
  if (top_candidates.size() > M_)
    throw std::runtime_error("Should not be more than M_ candidates returned by the heuristic");

  std::vector<tableint> selectedNeighbors;
  selectedNeighbors.reserve(M_);
  while (!top_candidates.empty()) {
    selectedNeighbors.push_back(top_candidates.top().second);
    top_candidates.pop();
  }
  tableint next_closest_entry_point = selectedNeighbors.back();

  {
    linklistsizeint* ll_cur;
    if (level == 0)
      ll_cur = reinterpret_cast<linklistsizeint*>(
          &data_level0_memory_[cur_c * size_data_per_element_]);
    else
      ll_cur = reinterpret_cast<linklistsizeint*>(
          link_lists_[cur_c] + (level - 1) * size_links_per_element_);

    if (*ll_cur) {
      throw std::runtime_error("The newly inserted element should have blank link list");
    }
    *ll_cur = static_cast<linklistsizeint>(selectedNeighbors.size());
    tableint* data = reinterpret_cast<tableint*>(ll_cur + 1);
    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
      if (data[idx])
        throw std::runtime_error("Possible memory corruption");
      if (level > element_levels_[selectedNeighbors[idx]])
        throw std::runtime_error("Trying to make a link on a non-existent level");
      data[idx] = selectedNeighbors[idx];
    }
  }

  for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
    linklistsizeint* ll_other;
    if (level == 0)
      ll_other = reinterpret_cast<linklistsizeint*>(
          &data_level0_memory_[selectedNeighbors[idx] * size_data_per_element_]);
    else
      ll_other = reinterpret_cast<linklistsizeint*>(
          link_lists_[selectedNeighbors[idx]] + (level - 1) * size_links_per_element_);

    size_t sz_link_list_other = *ll_other;

    if (sz_link_list_other > Mcurmax)
      throw std::runtime_error("Bad value of sz_link_list_other");
    if (selectedNeighbors[idx] == cur_c)
      throw std::runtime_error("Trying to connect an element to itself");
    if (level > element_levels_[selectedNeighbors[idx]])
      throw std::runtime_error("Trying to make a link on a non-existent level");

    tableint* data = reinterpret_cast<tableint*>(ll_other + 1);

    if (sz_link_list_other < Mcurmax) {
      data[sz_link_list_other] = cur_c;
      *ll_other = static_cast<linklistsizeint>(sz_link_list_other + 1);
    } else {
      float d_max = L2Sqr(data_point, GetDataByInternalId(selectedNeighbors[idx]), d_);
      std::priority_queue<std::pair<float, tableint>,
                          std::vector<std::pair<float, tableint>>,
                          FavorCompareByFirst<float>> candidates;
      candidates.emplace(d_max, cur_c);

      for (size_t j = 0; j < sz_link_list_other; j++) {
        candidates.emplace(
            L2Sqr(GetDataByInternalId(data[j]), GetDataByInternalId(selectedNeighbors[idx]), d_),
            data[j]);
      }

      GetNeighborsByHeuristic2(candidates, Mcurmax);

      int indx = 0;
      while (!candidates.empty()) {
        data[indx] = candidates.top().second;
        candidates.pop();
        indx++;
      }

      *ll_other = static_cast<linklistsizeint>(indx);
    }
  }

  return next_closest_entry_point;
}

void HakesFilterIndex::SearchBaseLayerFilter(
    tableint ep_id, const float* data_point, int level,
    std::priority_queue<std::pair<float, tableint>,
                        std::vector<std::pair<float, tableint>>,
                        FavorCompareByFirst<float>>& top_candidates) {
  std::vector<bool> visited(cur_element_count_, false);

  auto cmp_less = [](const std::pair<float, tableint>& a,
                     const std::pair<float, tableint>& b) {
    return a.first < b.first;
  };
  std::priority_queue<std::pair<float, tableint>,
                      std::vector<std::pair<float, tableint>>,
                      decltype(cmp_less)> candidate_set(cmp_less);

  float lowerBound;
  if (!IsMarkedDeleted(ep_id)) {
    float dist = L2Sqr(data_point, GetDataByInternalId(ep_id), d_);
    top_candidates.emplace(dist, ep_id);
    lowerBound = dist;
    candidate_set.emplace(-dist, ep_id);
  } else {
    lowerBound = std::numeric_limits<float>::max();
    candidate_set.emplace(-lowerBound, ep_id);
  }
  visited[ep_id] = true;

  while (!candidate_set.empty()) {
    auto curr_el_pair = candidate_set.top();
    if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
      break;
    }
    candidate_set.pop();

    tableint curNodeNum = curr_el_pair.second;

    int* data;
    if (level == 0)
      data = reinterpret_cast<int*>(
          &data_level0_memory_[curNodeNum * size_data_per_element_]);
    else
      data = reinterpret_cast<int*>(
          link_lists_[curNodeNum] + (level - 1) * size_links_per_element_);

    size_t size = *reinterpret_cast<linklistsizeint*>(data);
    tableint* datal = reinterpret_cast<tableint*>(data + 1);

    for (size_t j = 0; j < size; j++) {
      tableint candidate_id = datal[j];
      if (visited[candidate_id])
        continue;
      visited[candidate_id] = true;

      float dist1 = L2Sqr(data_point, GetDataByInternalId(candidate_id), d_);
      if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
        candidate_set.emplace(-dist1, candidate_id);
        if (!IsMarkedDeleted(candidate_id))
          top_candidates.emplace(dist1, candidate_id);
        if (top_candidates.size() > ef_construction_)
          top_candidates.pop();
        if (!top_candidates.empty())
          lowerBound = top_candidates.top().first;
      }
    }
  }
}

void HakesFilterIndex::SearchBaseLayerSTFilter(
    tableint ep_id, const float* query_data, 
    size_t ef, size_t k, float e_distance,
    const faiss::IDSelector* id_selector,
    std::vector<std::pair<float, tableint>>& results) const {
  
  std::vector<bool> visited;
  {
    std::unique_lock<std::mutex> lock(visited_pool_mutex_);
    if (!visited_pool_.empty()) {
      visited = std::move(visited_pool_.back());
      visited_pool_.pop_back();
    }
  }
  
  if (visited.size() < cur_element_count_) {
    visited.resize(cur_element_count_, false);
  }
  std::fill(visited.begin(), visited.begin() + cur_element_count_, false);
  
  auto cmp_greater = [](const std::pair<float, tableint>& a, 
                        const std::pair<float, tableint>& b) {
    return a.first > b.first;
  };
  auto cmp_less = [](const std::pair<float, tableint>& a, 
                     const std::pair<float, tableint>& b) {
    return a.first < b.first;
  };
  
  std::priority_queue<std::pair<float, tableint>, 
                      std::vector<std::pair<float, tableint>>,
                      decltype(cmp_greater)> candidate_set(cmp_greater);
  std::priority_queue<std::pair<float, tableint>, 
                      std::vector<std::pair<float, tableint>>,
                      decltype(cmp_less)> top_candidates(cmp_less);
  
  const float* ep_data = GetDataByInternalId(ep_id);
  float ep_dist = L2Sqr(query_data, ep_data, d_);
  
  bool ep_qualified = true;
  if (id_selector) {
    idx_t ep_label = GetExternalLabel(ep_id);
    ep_qualified = id_selector->is_member(ep_label);
  }
  
  float dist = ep_qualified ? ep_dist : ep_dist + e_distance;
  
  float lowerBound = dist;
  top_candidates.emplace(dist, ep_id);
  candidate_set.emplace(dist, ep_id);
  visited[ep_id] = true;
  
  size_t num_in_range = ep_qualified ? 1 : 0;
  
  while (!candidate_set.empty()) {
    auto current_node_pair = candidate_set.top();
    float candidate_dist = current_node_pair.first;
    
    if (candidate_dist > 0.95f * lowerBound && num_in_range > k * 0.5f) {
      break;
    }
    
    candidate_set.pop();
    tableint current_node_id = current_node_pair.second;
    
    unsigned int* data = reinterpret_cast<unsigned int*>(
        const_cast<char*>(&data_level0_memory_[current_node_id * size_data_per_element_]));
    int size = *reinterpret_cast<linklistsizeint*>(data);
    tableint* datal = reinterpret_cast<tableint*>(data + 1);
    
    for (int j = 0; j < size; j++) {
      tableint candidate_id = datal[j];
      
      if (visited[candidate_id]) {
        continue;
      }
      visited[candidate_id] = true;
      
      const float* currObj = GetDataByInternalId(candidate_id);
      
      bool candidate_qualified = true;
      if (id_selector) {
        idx_t candidate_label = GetExternalLabel(candidate_id);
        candidate_qualified = id_selector->is_member(candidate_label);
      }
      
      float dist1 = L2Sqr(query_data, currObj, d_);
      if (!candidate_qualified) {
        dist1 += e_distance;
      }
      
      if (top_candidates.size() < ef || lowerBound > dist1) {
        candidate_set.emplace(dist1, candidate_id);
        
        if (candidate_qualified) {
          num_in_range++;
        }
        
        top_candidates.emplace(dist1, candidate_id);
        
        if (top_candidates.size() > ef) {
          auto evicted = top_candidates.top();
          top_candidates.pop();
          
          idx_t evicted_label = GetExternalLabel(evicted.second);
          if (id_selector && id_selector->is_member(evicted_label)) {
            num_in_range--;
          }
        }
        
        if (!top_candidates.empty()) {
          lowerBound = top_candidates.top().first;
        }
      }
    }
  }
  
  {
    std::unique_lock<std::mutex> lock(visited_pool_mutex_);
    visited_pool_.push_back(std::move(visited));
  }
  
  results.clear();
  while (!top_candidates.empty()) {
    auto rez = top_candidates.top();
    top_candidates.pop();
    
    idx_t label = GetExternalLabel(rez.second);
    if (id_selector == nullptr || id_selector->is_member(label)) {
      results.emplace_back(rez.first, label);
    }
  }
  
  std::reverse(results.begin(), results.end());
}

void HakesFilterIndex::SearchGraphFilter(
    const float* query_data, size_t k, float p,
    const faiss::IDSelector* id_selector,
    std::vector<std::pair<float, idx_t>>& results) const {
  
  results.clear();
  if (cur_element_count_ == 0) {
    return;
  }
  
  float e_distance = DistFilter(p);
  
  tableint currObj = enterpoint_node_;
  float curdist = L2Sqr(query_data, GetDataByInternalId(currObj), d_);
  
  for (int level = maxlevel_; level > 0; level--) {
    bool changed = true;
    while (changed) {
      changed = false;
      unsigned int* data = reinterpret_cast<unsigned int*>(
          link_lists_[currObj] + (level - 1) * size_links_per_element_);
      int size = *reinterpret_cast<linklistsizeint*>(data);
      tableint* datal = reinterpret_cast<tableint*>(data + 1);
      
      for (int i = 0; i < size; i++) {
        tableint cand = datal[i];
        float d = L2Sqr(query_data, GetDataByInternalId(cand), d_);
        if (d < curdist) {
          curdist = d;
          currObj = cand;
          changed = true;
        }
      }
    }
  }
  
  size_t ef = std::max(ef_, k);
  std::vector<std::pair<float, tableint>> top_candidates;
  SearchBaseLayerSTFilter(currObj, query_data, ef, k, 
                          e_distance / static_cast<float>(ef), 
                          id_selector, top_candidates);
  
  for (const auto& candidate : top_candidates) {
    results.push_back(candidate);
    if (results.size() >= k) {
      break;
    }
  }
}

void HakesFilterIndex::SearchBruteForceFilter(
    const float* query_data, size_t k,
    const faiss::IDSelector* id_selector,
    std::vector<std::pair<float, idx_t>>& results) const {
  
  results.clear();
  
  auto cmp = [](const std::pair<float, idx_t>& a, const std::pair<float, idx_t>& b) {
    return a.first < b.first;
  };
  std::priority_queue<std::pair<float, idx_t>, 
                      std::vector<std::pair<float, idx_t>>,
                      decltype(cmp)> result_heap(cmp);
  
  for (size_t i = 0; i < cur_element_count_; i++) {
    idx_t label = GetExternalLabel(static_cast<tableint>(i));
    
    if (id_selector && !id_selector->is_member(label)) {
      continue;
    }
    
    float dist = L2Sqr(query_data, GetDataByInternalId(static_cast<tableint>(i)), d_);
    
    if (result_heap.size() < k) {
      result_heap.emplace(dist, label);
    } else if (dist < result_heap.top().first) {
      result_heap.emplace(dist, label);
      result_heap.pop();
    }
  }
  
  while (!result_heap.empty()) {
    results.push_back(result_heap.top());
    result_heap.pop();
  }
  
  std::reverse(results.begin(), results.end());
}

bool HakesFilterIndex::SaveFilterIndex(const std::string& path) const {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }
  
  file.write(reinterpret_cast<const char*>(&d_), sizeof(d_));
  file.write(reinterpret_cast<const char*>(&max_elements_), sizeof(max_elements_));
  file.write(reinterpret_cast<const char*>(&cur_element_count_), sizeof(cur_element_count_));
  file.write(reinterpret_cast<const char*>(&M_), sizeof(M_));
  file.write(reinterpret_cast<const char*>(&maxM_), sizeof(maxM_));
  file.write(reinterpret_cast<const char*>(&maxM0_), sizeof(maxM0_));
  file.write(reinterpret_cast<const char*>(&ef_construction_), sizeof(ef_construction_));
  file.write(reinterpret_cast<const char*>(&ef_), sizeof(ef_));
  file.write(reinterpret_cast<const char*>(&delta_d_), sizeof(delta_d_));
  file.write(reinterpret_cast<const char*>(&mult_), sizeof(mult_));
  file.write(reinterpret_cast<const char*>(&enterpoint_node_), sizeof(enterpoint_node_));
  file.write(reinterpret_cast<const char*>(&maxlevel_), sizeof(maxlevel_));
  
  file.write(reinterpret_cast<const char*>(&size_links_level0_), sizeof(size_links_level0_));
  file.write(reinterpret_cast<const char*>(&size_data_per_element_), sizeof(size_data_per_element_));
  file.write(reinterpret_cast<const char*>(&offset_data_), sizeof(offset_data_));
  file.write(reinterpret_cast<const char*>(&label_offset_), sizeof(label_offset_));
  file.write(reinterpret_cast<const char*>(&offset_level0_), sizeof(offset_level0_));
  file.write(reinterpret_cast<const char*>(&size_links_per_element_), sizeof(size_links_per_element_));
  
  file.write(data_level0_memory_.data(), 
             static_cast<std::streamsize>(cur_element_count_ * size_data_per_element_));
  
  file.write(reinterpret_cast<const char*>(element_levels_.data()),
             static_cast<std::streamsize>(cur_element_count_ * sizeof(int)));
  
  for (size_t i = 0; i < cur_element_count_; i++) {
    int level = element_levels_[i];
    if (level > 0) {
      unsigned int link_size = level * static_cast<unsigned int>(size_links_per_element_);
      file.write(reinterpret_cast<const char*>(&link_size), sizeof(link_size));
      file.write(link_lists_[i], link_size);
    } else {
      unsigned int link_size = 0;
      file.write(reinterpret_cast<const char*>(&link_size), sizeof(link_size));
    }
  }
  
  size_t label_count = label_lookup_.size();
  file.write(reinterpret_cast<const char*>(&label_count), sizeof(label_count));
  for (const auto& pair : label_lookup_) {
    file.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
    file.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
  }
  
  size_t deleted_count = deleted_elements_.size();
  file.write(reinterpret_cast<const char*>(&deleted_count), sizeof(deleted_count));
  for (tableint id : deleted_elements_) {
    file.write(reinterpret_cast<const char*>(&id), sizeof(id));
  }
  
  file.close();
  return true;
}

bool HakesFilterIndex::LoadFilterIndex(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }
  
  for (size_t i = 0; i < link_lists_.size(); i++) {
    if (link_lists_[i] != nullptr) {
      free(link_lists_[i]);
    }
  }
  link_lists_.clear();
  
  file.read(reinterpret_cast<char*>(&d_), sizeof(d_));
  file.read(reinterpret_cast<char*>(&max_elements_), sizeof(max_elements_));
  file.read(reinterpret_cast<char*>(&cur_element_count_), sizeof(cur_element_count_));
  file.read(reinterpret_cast<char*>(&M_), sizeof(M_));
  file.read(reinterpret_cast<char*>(&maxM_), sizeof(maxM_));
  file.read(reinterpret_cast<char*>(&maxM0_), sizeof(maxM0_));
  file.read(reinterpret_cast<char*>(&ef_construction_), sizeof(ef_construction_));
  file.read(reinterpret_cast<char*>(&ef_), sizeof(ef_));
  file.read(reinterpret_cast<char*>(&delta_d_), sizeof(delta_d_));
  file.read(reinterpret_cast<char*>(&mult_), sizeof(mult_));
  file.read(reinterpret_cast<char*>(&enterpoint_node_), sizeof(enterpoint_node_));
  file.read(reinterpret_cast<char*>(&maxlevel_), sizeof(maxlevel_));
  
  file.read(reinterpret_cast<char*>(&size_links_level0_), sizeof(size_links_level0_));
  file.read(reinterpret_cast<char*>(&size_data_per_element_), sizeof(size_data_per_element_));
  file.read(reinterpret_cast<char*>(&offset_data_), sizeof(offset_data_));
  file.read(reinterpret_cast<char*>(&label_offset_), sizeof(label_offset_));
  file.read(reinterpret_cast<char*>(&offset_level0_), sizeof(offset_level0_));
  file.read(reinterpret_cast<char*>(&size_links_per_element_), sizeof(size_links_per_element_));
  
  data_level0_memory_.resize(max_elements_ * size_data_per_element_);
  file.read(data_level0_memory_.data(),
            static_cast<std::streamsize>(cur_element_count_ * size_data_per_element_));
  
  element_levels_.resize(max_elements_);
  file.read(reinterpret_cast<char*>(element_levels_.data()),
            static_cast<std::streamsize>(cur_element_count_ * sizeof(int)));
  
  link_lists_.resize(max_elements_, nullptr);
  link_list_locks_.clear();
  link_list_locks_.reserve(max_elements_);
  for (size_t i = 0; i < max_elements_; i++) {
    link_list_locks_.emplace_back(std::make_unique<std::mutex>());
  }
  
  for (size_t i = 0; i < cur_element_count_; i++) {
    unsigned int link_size;
    file.read(reinterpret_cast<char*>(&link_size), sizeof(link_size));
    if (link_size > 0) {
      link_lists_[i] = static_cast<char*>(malloc(link_size));
      file.read(link_lists_[i], link_size);
    }
  }
  
  size_t label_count;
  file.read(reinterpret_cast<char*>(&label_count), sizeof(label_count));
  label_lookup_.clear();
  for (size_t i = 0; i < label_count; i++) {
    idx_t label;
    tableint internal_id;
    file.read(reinterpret_cast<char*>(&label), sizeof(label));
    file.read(reinterpret_cast<char*>(&internal_id), sizeof(internal_id));
    label_lookup_[label] = internal_id;
  }
  
  size_t deleted_count;
  file.read(reinterpret_cast<char*>(&deleted_count), sizeof(deleted_count));
  deleted_elements_.clear();
  for (size_t i = 0; i < deleted_count; i++) {
    tableint id;
    file.read(reinterpret_cast<char*>(&id), sizeof(id));
    deleted_elements_.insert(id);
  }
  
  ntotal_ = static_cast<idx_t>(cur_element_count_);
  
  file.close();
  return true;
}

}  // namespace faiss
