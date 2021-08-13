use rust_bert::pipelines::question_answering::{QuestionAnsweringModel, QaInput};
use anyhow;

fn main() -> anyhow::Result<()> {
    let qa_model = QuestionAnsweringModel::new(Default::default())?;
    let question = String::from("Where does Amy live ?");
    let context = String::from("Amy lives in Amsterdam");

    let answers = qa_model.predict(&[QaInput { question, context }], 1, 32);
    println!("{:?}", answers);
    Ok(())
}
