// Optimized state memory cache with concurrency, LRU, and trait-separated key design.

use alloy_primitives::{Address, B256, U256};
use revm_database::states::{PlainStorageChangeset, StateChangeset};
use reth_primitives_traits::{Account, Bytecode};
use reth_trie_common::BranchNodeCompact;
use moka::sync::Cache;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use tracing::{debug, info};

#[derive(Clone, Copy, Debug)]
pub enum CacheMode {
    ReadOnly,
    ReadWrite,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone)]
enum Key {
    Addr(Address),
    CodeHash(B256),
    StorageSlot(B256),
    TrieAccount(Vec<u8>),
    TrieStorage(B256, Vec<u8>),
}

#[derive(Clone, Debug)]
enum Value {
    Account(Account),
    Bytecode(Bytecode),
    Storage(U256),
    TrieNode(BranchNodeCompact),
}

/// A thread-safe LRU cache wrapper with optional read-only mode.
#[derive(Debug, Clone)]
pub struct ConsistentMemory<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + std::fmt::Debug + 'static,
    V: Clone + Send + Sync + std::fmt::Debug + 'static,
{
    inner: Cache<K, V>,
    mode: CacheMode,
}

impl<K, V> ConsistentMemory<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + std::fmt::Debug,
    V: Clone + Send + Sync + std::fmt::Debug,
{
    pub fn new(capacity: u64, mode: CacheMode) -> Self {
        let inner = Cache::builder().max_capacity(capacity).build();
        Self { inner, mode }
    }

    pub fn insert(&self, key: K, value: V) {
        if matches!(self.mode, CacheMode::ReadWrite) {
            self.inner.insert(key, value);
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.get(key)
    }

    pub fn remove(&self, key: &K) {
        if matches!(self.mode, CacheMode::ReadWrite) {
            self.inner.invalidate(key);
        }
    }

    pub fn clear(&self) {
        self.inner.invalidate_all();
    }
}

#[derive(Debug, Clone)]
pub struct StateMemory {
    capacity: u64,
    accounts: ConsistentMemory<Key, Value>,
    storages: ConsistentMemory<Key, Arc<ConsistentMemory<Key, Value>>>,
    bytecodes: ConsistentMemory<Key, Value>,
}


#[derive(Debug, Clone)]
struct MerkleMemory {
    capacity: u64,
    inner: ConsistentMemory<Key, Value>,
}


// TODO(brain@lazai): should add the feature of multi-version memory to global_latest_memory
#[derive(Debug)]
pub struct GlobalConsistentMemory {
    inner: RwLock<GlobalConsistentMemoryInner>,
}

#[derive(Debug)]
struct GlobalConsistentMemoryInner {
    pub latest_block_number: u64,

    pub(crate) state_memory: StateMemory,

    pub(crate) merkle_memory: MerkleMemory,
}

impl Clone for GlobalConsistentMemory {
    fn clone(&self) -> Self {
        let mut inner = self.inner.read().unwrap();
        Self {
            inner: RwLock::new(GlobalConsistentMemoryInner{
                latest_block_number: inner.latest_block_number,
                state_memory: inner.state_memory.clone(),
                merkle_memory: inner.merkle_memory.clone(),
            }),
        }
    }
}

impl StateMemory {
    pub fn new(capacity: u64, mode: CacheMode) -> Self {
        Self {
            capacity,
            accounts: ConsistentMemory::new(capacity, mode),
            storages: ConsistentMemory::new(capacity, mode),
            bytecodes: ConsistentMemory::new(capacity, mode),
        }
    }

    pub fn get_account(&self, addr: &Address) -> Option<Account> {
        self.accounts.get(&Key::Addr(*addr)).and_then(|v| match v {
            Value::Account(a) => Some(a),
            _ => None,
        })
    }

    pub fn insert_account(&self, addr: Address, acc: Account) {
        self.accounts.insert(Key::Addr(addr), Value::Account(acc));
    }

    pub fn remove_account(&self, addr: Address) {
        self.accounts.remove(&Key::Addr(addr));
        self.storages.remove(&Key::Addr(addr));
    }

    pub fn get_bytecode(&self, code_hash: &B256) -> Option<Bytecode> {
        self.bytecodes.get(&Key::CodeHash(*code_hash)).and_then(|v| match v {
            Value::Bytecode(b) => Some(b),
            _ => None,
        })
    }

    pub fn insert_code(&self, hash: B256, code: Bytecode) {
        self.bytecodes.insert(Key::CodeHash(hash), Value::Bytecode(code));
    }

    pub fn remove_code(&self, hash: B256) {
        self.bytecodes.remove(&Key::CodeHash(hash));
    }

    pub fn get_storage(&self, addr: &Address, slot: &B256) -> Option<U256> {
        self.storages
            .get(&Key::Addr(*addr))
            .and_then(|inner| inner.get(&Key::StorageSlot(*slot)))
            .and_then(|v| match v {
                Value::Storage(u) => Some(u),
                _ => None,
            })
    }

    pub fn insert_storage(&self, addr: Address, slot: B256, val: U256) {
        let key = Key::Addr(addr);
        let slot_key = Key::StorageSlot(slot);
        let storage = self.storages.get(&key).unwrap_or_else(|| {
            let inner = Arc::new(ConsistentMemory::new(self.capacity, CacheMode::ReadWrite));
            self.storages.insert(key.clone(), inner.clone());
            inner
        });
        storage.insert(slot_key, Value::Storage(val));
    }

    pub fn remove_storage(&self, addr: Address, slot: B256) {
        let key = Key::Addr(addr);
        let slot_key = Key::StorageSlot(slot);
        if let Some(inner) = self.storages.get(&key) {
            inner.remove(&slot_key);
        }
    }

    pub fn clear(&self) {
        self.accounts.clear();
        self.storages.clear();
        self.bytecodes.clear();
    }
}

impl MerkleMemory {
    fn new(capacity: u64, mode: CacheMode) -> Self {
        Self {
            capacity: capacity,
            inner: ConsistentMemory::new(capacity, mode)
        }
    }
    fn get_trie_account(&self, nibbles: &Vec<u8>) -> Option<BranchNodeCompact> {
        match self.inner.get(&Key::TrieAccount(nibbles.clone())) {
            Some(Value::TrieNode(n)) => Some(n),
            _ => None,
        }
    }

    fn get_trie_storage(&self, addr_hash: &B256, nibbles: &Vec<u8>) -> Option<BranchNodeCompact> {
        match self.inner.get(&Key::TrieStorage(*addr_hash, nibbles.clone())) {
            Some(Value::TrieNode(n)) => Some(n),
            _ => None,
        }
    }

    fn insert_trie_node(&self, hash: B256, nibbles: Vec<u8>, node: BranchNodeCompact, is_account: bool) {
        let key = if is_account {
            Key::TrieAccount(nibbles)
        } else {
            Key::TrieStorage(hash, nibbles)
        };
        self.inner.insert(key, Value::TrieNode(node));
    }

    fn clear(&self) {
        self.inner.clear();
    }
}




// Optional helper: insert functions for ParallelEvmCache types
impl GlobalConsistentMemoryInner {
    pub fn insert_account(&self, addr: Address, acc: Account) {
        self.state_memory.insert_account(addr, acc);
    }

    pub fn remove_account(&self, addr: Address) {
        self.state_memory.remove_account(addr);
    }

    pub fn clear_all_account_storage(&self, addr: Address) {
        self.state_memory.remove_account(addr);
    }

    pub fn insert_code(&self, hash: B256, code: Bytecode) {
        self.state_memory.insert_code(hash, code);
    }

    pub fn remove_code(&self, hash: B256) {
        self.state_memory.remove_code(hash);
    }

    pub fn insert_storage(&self, addr: Address, slot: B256, val: U256) {
        self.state_memory.insert_storage(addr, slot, val);
    }

    pub fn remove_storage(&self, addr: Address, slot: B256) {
        self.state_memory.remove_storage(addr, slot);
    }

    pub fn insert_trie_node(&self, hash: B256, nibbles: Vec<u8>, node: BranchNodeCompact, is_account: bool) {
        self.merkle_memory.insert_trie_node(hash, nibbles, node, is_account);
    }

    pub fn clear(&mut self) {
        self.latest_block_number = u64::MAX - 1;
        self.merkle_memory.clear();
        self.state_memory.clear();
    }
}



// Optional helper: insert functions for ParallelEvmCache types
impl GlobalConsistentMemory {
    pub fn new(state_capacity: u64, merkle_capacity: u64, mode: CacheMode) -> Self {
        Self {
            inner: RwLock::new(GlobalConsistentMemoryInner {
                latest_block_number: u64::MAX - 1,
                state_memory: StateMemory::new(state_capacity, mode),
                merkle_memory: MerkleMemory::new(merkle_capacity, mode),
            })
        }
    }

    pub fn update_global_memory(&self, block_number: u64, state_changeset: StateChangeset) {
        let mut inner = self.inner.write().unwrap();
        if inner.latest_block_number != u64::MAX - 1 && block_number != inner.latest_block_number + 1 {
            info!("Cannot Update Global Memory latest_block_number={}, target block_number={}", inner.latest_block_number, block_number);
            panic!("Invalid Block number and cannot update global memory");
        }
        info!("Update Global Memory latest_block_number={}, target block_number={}", inner.latest_block_number, block_number);


        for (address, account) in state_changeset.accounts {
            inner.insert_account(address, account.unwrap().into());
        }

        for (hash, bytecode) in state_changeset.contracts {
            inner.insert_code(hash, Bytecode(bytecode));
        }

        for PlainStorageChangeset { address, wipe_storage, storage } in state_changeset.storage {
            if wipe_storage {
                inner.remove_account(address);
            } else {
                // Ignore all storages which wiped
                for (slot, value) in storage {
                    inner.insert_storage(address, slot.into(), value);
                }
            }
        }
        inner.latest_block_number = block_number;
    }


    pub fn get_account(&self, addr: &Address) -> Option<Account> {
        self.inner.read().unwrap().state_memory.get_account(addr)
    }

    pub fn get_bytecode(&self, code_hash: &B256) -> Option<Bytecode> {
        self.inner.read().unwrap().state_memory.get_bytecode(code_hash)
    }

    pub fn get_storage(&self, addr: &Address, slot: &B256) -> Option<U256> {
        self.inner.read().unwrap().state_memory.get_storage(addr, slot)
    }

    pub fn clear(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.clear();
    }
}
