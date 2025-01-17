/// The module responsible to work with SQLite database

#[path = "database/my_swaps.rs"]
pub mod my_swaps;
#[path = "database/stats_swaps.rs"] pub mod stats_swaps;

use crate::CREATE_MY_SWAPS_TABLE;
use common::{log::{debug, error, info},
             mm_ctx::MmArc,
             rusqlite::{Connection, Result as SqlResult, NO_PARAMS}};

use my_swaps::fill_my_swaps_from_json_statements;
use stats_swaps::create_and_fill_stats_swaps_from_json_statements;

const SELECT_MIGRATION: &str = "SELECT * FROM migration ORDER BY current_migration DESC LIMIT 1;";

fn get_current_migration(conn: &Connection) -> SqlResult<i64> {
    conn.query_row(SELECT_MIGRATION, NO_PARAMS, |row| row.get(0))
}

pub fn init_and_migrate_db(ctx: &MmArc, conn: &Connection) -> SqlResult<()> {
    info!("Checking the current SQLite migration");
    match get_current_migration(conn) {
        Ok(current_migration) => {
            if current_migration >= 1 {
                info!(
                    "Current migration is {}, skipping the init, trying to migrate",
                    current_migration
                );
                migrate_sqlite_database(ctx, conn, current_migration)?;
                return Ok(());
            }
        },
        Err(e) => {
            debug!("Error {} on getting current migration. The database is either empty or corrupted, trying to clean it first", e);
            if let Err(e) = conn.execute_batch(
                "DROP TABLE migration;
                    DROP TABLE my_swaps;",
            ) {
                error!("Error {} on SQLite database cleanup", e);
            }
        },
    };

    info!("Trying to initialize the SQLite database");

    let init_batch = concat!(
        "BEGIN;
        CREATE TABLE IF NOT EXISTS migration (current_migration INTEGER NOT_NULL UNIQUE);
        INSERT INTO migration (current_migration) VALUES (1);",
        CREATE_MY_SWAPS_TABLE!(),
        "COMMIT;"
    );
    conn.execute_batch(init_batch)?;
    migrate_sqlite_database(ctx, conn, 1)?;
    info!("SQLite database initialization is successful");
    Ok(())
}

fn migration_1(ctx: &MmArc) -> Vec<(&'static str, Vec<String>)> { fill_my_swaps_from_json_statements(ctx) }

fn migration_2(ctx: &MmArc) -> Vec<(&'static str, Vec<String>)> {
    create_and_fill_stats_swaps_from_json_statements(ctx)
}

fn migration_3() -> Vec<(&'static str, Vec<String>)> { vec![(stats_swaps::ADD_STARTED_AT_INDEX, vec![])] }

fn migration_4() -> Vec<(&'static str, Vec<String>)> { stats_swaps::add_and_split_tickers() }

fn statements_for_migration(ctx: &MmArc, current_migration: i64) -> Option<Vec<(&'static str, Vec<String>)>> {
    match current_migration {
        1 => Some(migration_1(ctx)),
        2 => Some(migration_2(ctx)),
        3 => Some(migration_3()),
        4 => Some(migration_4()),
        _ => None,
    }
}

pub fn migrate_sqlite_database(ctx: &MmArc, conn: &Connection, mut current_migration: i64) -> SqlResult<()> {
    info!("migrate_sqlite_database, current migration {}", current_migration);
    let transaction = conn.unchecked_transaction()?;
    while let Some(statements_with_params) = statements_for_migration(ctx, current_migration) {
        for (statement, params) in statements_with_params {
            debug!("Executing SQL statement {:?} with params {:?}", statement, params);
            transaction.execute(&statement, params)?;
        }
        current_migration += 1;
        transaction.execute("INSERT INTO migration (current_migration) VALUES (?1);", &[
            current_migration,
        ])?;
    }
    transaction.commit()?;
    info!("migrate_sqlite_database complete, migrated to {}", current_migration);
    Ok(())
}
